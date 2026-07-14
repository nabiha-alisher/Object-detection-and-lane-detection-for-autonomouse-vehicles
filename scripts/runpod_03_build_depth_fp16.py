import os, sys, json, importlib, shutil
import cv2, torch, onnx, tensorrt as trt

PROJECT=os.environ.get('PROJECT','/workspace/fyp')
paths=json.load(open(os.path.join(PROJECT,'runpod_paths.json')))
METRIC_ROOT=paths['metric_root']
DEPTH_PT=paths['depth_pt']
VIDEO=paths.get('video')
ONNX_DIR=os.path.join(PROJECT,'onnx'); ENG_DIR=os.path.join(PROJECT,'engines')
os.makedirs(ONNX_DIR,exist_ok=True); os.makedirs(ENG_DIR,exist_ok=True)

assert os.path.isdir(METRIC_ROOT), METRIC_ROOT
assert os.path.exists(DEPTH_PT), DEPTH_PT
if METRIC_ROOT in sys.path: sys.path.remove(METRIC_ROOT)
sys.path.insert(0,METRIC_ROOT)
for k in list(sys.modules.keys()):
    if k=='depth_anything_v2' or k.startswith('depth_anything_v2.'):
        del sys.modules[k]
importlib.invalidate_caches()
from depth_anything_v2.dpt import DepthAnythingV2

device='cuda'; torch.backends.cudnn.benchmark=True
cfg={'encoder':'vits','features':64,'out_channels':[48,96,192,384]}
metric_model=DepthAnythingV2(**cfg,max_depth=80)
ckpt=torch.load(DEPTH_PT,map_location='cpu')
if isinstance(ckpt,dict) and 'state_dict' in ckpt: ckpt=ckpt['state_dict']
elif isinstance(ckpt,dict) and 'model' in ckpt: ckpt=ckpt['model']
if isinstance(ckpt,dict): ckpt={k.replace('module.','',1):v for k,v in ckpt.items()}
missing,unexpected=metric_model.load_state_dict(ckpt,strict=False)
print('Missing keys:',len(missing),'Unexpected:',len(unexpected))
metric_model=metric_model.to(device).eval()

if VIDEO and os.path.exists(VIDEO):
    cap=cv2.VideoCapture(VIDEO); ok,raw_img=cap.read(); cap.release(); assert ok
else:
    raw_img = (torch.zeros(518, 518, 3).numpy()).astype('uint8')

dummy,(proc_h,proc_w)=metric_model.image2tensor(raw_img,input_size=518)
dummy=dummy.to(device)
print('Dummy:',tuple(dummy.shape),'processed:',proc_h,proc_w)

class MetricDepthExport(torch.nn.Module):
    def __init__(self,m): super().__init__(); self.model=m
    def forward(self,x): return self.model.forward(x)

ONNX_PATH=os.path.join(ONNX_DIR,'depth_anything_v2_metric_vkitti_vits_518.onnx')
ENGINE_PATH=os.path.join(ENG_DIR,'depth_anything_v2_metric_vkitti_vits_fp16.engine')
if not os.path.exists(ONNX_PATH):
    torch.onnx.export(MetricDepthExport(metric_model).eval(),dummy,ONNX_PATH,input_names=['images'],output_names=['depth'],opset_version=17,do_constant_folding=True,dynamic_axes=None)
    m=onnx.load(ONNX_PATH); onnx.checker.check_model(m)
    print('Saved ONNX:',ONNX_PATH)
else:
    print('Using existing ONNX:',ONNX_PATH)

assert str(trt.__version__).startswith('10.0.1'), trt.__version__
TRT_LOGGER=trt.Logger(trt.Logger.WARNING)
if os.path.exists(ENGINE_PATH): os.remove(ENGINE_PATH)
N,C,H,W=tuple(dummy.shape)
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network,TRT_LOGGER) as parser:
    ok=parser.parse(open(ONNX_PATH,'rb').read())
    if not ok:
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config(); config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,4*(1<<30)); config.set_flag(trt.BuilderFlag.FP16)
    profile=builder.create_optimization_profile(); profile.set_shape('images',(N,C,H,W),(N,C,H,W),(N,C,H,W)); config.add_optimization_profile(profile)
    plan=builder.build_serialized_network(network,config)
    if plan is None: raise RuntimeError('build_serialized_network returned None')
    open(ENGINE_PATH,'wb').write(plan)
print('Saved depth engine:',ENGINE_PATH)
