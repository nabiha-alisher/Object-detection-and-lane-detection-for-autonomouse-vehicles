import os, sys, types, importlib, random, json, shutil, gc
import numpy as np
import torch
import torchvision.transforms as transforms
import tensorrt as trt
import pycuda.driver as cuda

PROJECT = os.environ.get('PROJECT', '/workspace/fyp')
paths = json.load(open(os.path.join(PROJECT, 'runpod_paths.json')))
YOLOPX_ROOT = paths['yolopx_root']
WEIGHTS = paths['checkpoint']
VAL_BASE = paths['val_base']
OUT_DIR = os.path.join(PROJECT, 'engines')
ONNX_DIR = os.path.join(PROJECT, 'onnx')
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(ONNX_DIR, exist_ok=True)

N_CALIB = int(os.environ.get('N_CALIB', '2500'))
WORKSPACE_MB = int(os.environ.get('WORKSPACE_MB', '4096'))
CONF_THRES = 0.3
IOU_THRES = 0.6

device = torch.device('cuda')
torch.zeros(1, device=device)
print('GPU:', torch.cuda.get_device_name(0))
print('TRT:', trt.__version__)
assert str(trt.__version__).startswith('10.0.1'), trt.__version__

cuda.init(); _ctx = cuda.Device(0).retain_primary_context(); _ctx.push()

os.chdir(YOLOPX_ROOT)
while YOLOPX_ROOT in sys.path: sys.path.remove(YOLOPX_ROOT)
sys.path.insert(0, YOLOPX_ROOT)
for k in list(sys.modules.keys()):
    if k == 'lib' or k.startswith('lib.'):
        del sys.modules[k]
lib_pkg = types.ModuleType('lib'); lib_pkg.__path__ = [os.path.join(YOLOPX_ROOT, 'lib')]; sys.modules['lib'] = lib_pkg
importlib.invalidate_caches()

from lib.config import cfg
from lib.models import get_net
import lib.dataset as dataset
from lib.core.evaluate import SegmentationMetric
from lib.core.general import non_max_suppression, scale_coords, xywh2xyxy, box_iou, ap_per_class
try:
    from lib.utils import DataLoaderX as LoaderClass
except Exception:
    from torch.utils.data import DataLoader as LoaderClass

cfg.WORKERS = 0
cfg.TEST.BATCH_SIZE_PER_GPU = 1
cfg.DATASET.DATAROOT  = os.path.join(VAL_BASE, 'imagess', 'val')
cfg.DATASET.LABELROOT = os.path.join(VAL_BASE, 'labelss', 'val')
cfg.DATASET.MASKROOT  = os.path.join(VAL_BASE, 'damaskss', 'val')
cfg.DATASET.LANEROOT  = os.path.join(VAL_BASE, 'lanemasks', 'val')
cfg.DATASET.DATASET   = 'BddDataset'
for p in [cfg.DATASET.DATAROOT, cfg.DATASET.LABELROOT, cfg.DATASET.MASKROOT, cfg.DATASET.LANEROOT]:
    assert os.path.isdir(p), p

normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
val_tf = transforms.Compose([transforms.ToTensor(), normalize])

# ===== RUNPOD VALSET REAL PATH FIX =====
# Actual unpacked valset structure:
# images  -> /workspace/fyp/data/valset/valset/imagess/val
# labels  -> /workspace/fyp/data/valset/valset/labelss/val
# DA mask -> /workspace/fyp/data/valset/valset/damaskss/val
# LL mask -> /workspace/fyp/data/valset/valset/lanemasks/val
if hasattr(cfg, "defrost"):
    cfg.defrost()

cfg.DATASET.DATAROOT  = "/workspace/fyp/data/valset/valset/imagess"
cfg.DATASET.LABELROOT = "/workspace/fyp/data/valset/valset/labelss"
cfg.DATASET.MASKROOT  = "/workspace/fyp/data/valset/valset/damaskss"
cfg.DATASET.LANEROOT  = "/workspace/fyp/data/valset/valset/lanemasks"

if hasattr(cfg, "freeze"):
    cfg.freeze()

print("FORCED DATASET ROOTS:")
print("  DATAROOT :", cfg.DATASET.DATAROOT)
print("  LABELROOT:", cfg.DATASET.LABELROOT)
print("  MASKROOT :", cfg.DATASET.MASKROOT)
print("  LANEROOT :", cfg.DATASET.LANEROOT)
# ===== END RUNPOD VALSET REAL PATH FIX =====

ds = dataset.BddDataset(cfg=cfg, is_train=False, inputsize=cfg.MODEL.IMAGE_SIZE, transform=val_tf)
loader = LoaderClass(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, collate_fn=dataset.AutoDriveDataset.collate_fn)
sample_img, *_ = ds[0]
H, W = int(sample_img.shape[1]), int(sample_img.shape[2])
print('Validation tensor shape:', (1,3,H,W))
assert (H,W) == (384,640), f'Expected final YOLOPX shape (384,640), got {(H,W)}'

ONNX_LOCAL = os.path.join(ONNX_DIR, f'yolopx_fp32_{H}x{W}.onnx')
ENGINE_INT8 = os.path.join(OUT_DIR, f'yolopx_int8_{H}x{W}.engine')
CACHE_FILE = os.path.join(OUT_DIR, f'yolopx_int8_{H}x{W}_calib{N_CALIB}.cache')

# Use uploaded ONNX if available; otherwise export from checkpoint exactly like final notebook.
if not os.path.exists(ONNX_LOCAL):
    model = get_net(cfg)
    ckpt = torch.load(WEIGHTS, map_location='cpu')
    sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    md = model.state_dict(); md.update(sd); model.load_state_dict(md, strict=True)
    model.eval().to(device).float()
    class ExportWrap(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, images):
            det_out, da_seg, ll_seg = self.m(images)
            inf_out, _ = det_out
            return inf_out, da_seg, ll_seg
    wrap = ExportWrap(model).eval().to(device)
    dummy = torch.randn(1,3,H,W, device=device)
    torch.onnx.export(wrap, dummy, ONNX_LOCAL, input_names=['images'], output_names=['det_out','da_seg','ll_seg'], opset_version=18, do_constant_folding=True, dynamo=False, export_params=True)
    sz = os.path.getsize(ONNX_LOCAL)/(1024*1024)
    print('Saved ONNX:', ONNX_LOCAL, 'MB=', sz)
    assert sz > 5, 'ONNX too small; export failed'
    del model, wrap, dummy; torch.cuda.empty_cache(); gc.collect()
else:
    print('Using existing ONNX:', ONNX_LOCAL)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
class DatasetEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, ds, n_calib, cache_file):
        super().__init__(); self.ds=ds; self.cache_file=cache_file
        idxs=list(range(len(ds))); random.seed(0); random.shuffle(idxs)
        self.idxs=idxs[:min(n_calib,len(ds))]; self.i=0
        c,h,w=ds[0][0].shape; self.shape=(1,c,h,w)
        self.d_input=cuda.mem_alloc(int(np.prod(self.shape)*4))
    def get_batch_size(self): return 1
    def get_batch(self, names):
        if self.i >= len(self.idxs): return None
        img,*_=self.ds[self.idxs[self.i]]; self.i += 1
        x=img.unsqueeze(0).contiguous().numpy().astype(np.float32)
        cuda.memcpy_htod(self.d_input, x)
        return [int(self.d_input)]
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file): return open(self.cache_file,'rb').read()
        return None
    def write_calibration_cache(self, cache): open(self.cache_file,'wb').write(cache)

if os.path.exists(ENGINE_INT8): os.remove(ENGINE_INT8)
if os.path.exists(CACHE_FILE) and os.environ.get('FORCE_RECALIB_CACHE','0') == '1': os.remove(CACHE_FILE)
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    assert parser.parse(open(ONNX_LOCAL,'rb').read()), '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
    config=builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_MB*(1<<20))
    profile=builder.create_optimization_profile(); profile.set_shape('images',(1,3,H,W),(1,3,H,W),(1,3,H,W)); config.add_optimization_profile(profile)
    config.set_flag(trt.BuilderFlag.INT8); config.int8_calibrator=DatasetEntropyCalibrator(ds,N_CALIB,CACHE_FILE)
    if builder.platform_has_fast_fp16: config.set_flag(trt.BuilderFlag.FP16)
    plan=builder.build_serialized_network(network, config)
    if plan is None: raise RuntimeError('build_serialized_network failed')
    open(ENGINE_INT8,'wb').write(plan)
print('Saved YOLOPX engine:', ENGINE_INT8)
print('Saved cache:', CACHE_FILE)

# Verify I/O names/shapes
with trt.Runtime(TRT_LOGGER) as rt:
    eng=rt.deserialize_cuda_engine(open(ENGINE_INT8,'rb').read())
ctx=eng.create_execution_context(); ctx.set_input_shape('images',(1,3,H,W))
for i in range(eng.num_io_tensors):
    n=eng.get_tensor_name(i); print(n, eng.get_tensor_mode(n), tuple(ctx.get_tensor_shape(n)), eng.get_tensor_dtype(n))
need={'images','det_out','da_seg','ll_seg'}
have={eng.get_tensor_name(i) for i in range(eng.num_io_tensors)}
assert need.issubset(have), have
