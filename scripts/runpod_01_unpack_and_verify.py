import os, zipfile, shutil, glob, json, sys

PROJECT = os.environ.get('PROJECT', '/workspace/fyp')
UPLOAD = '/workspace/upload/RUNPOD_REQUIRED_3MODEL_REBUILD.zip'
INROOT = os.path.join(PROJECT, 'input')
SRC = os.path.join(PROJECT, 'src')
DATA = os.path.join(PROJECT, 'data')
ENG = os.path.join(PROJECT, 'engines')
ONNX = os.path.join(PROJECT, 'onnx')
ASSETS = os.path.join(PROJECT, 'assets')

assert os.path.exists(UPLOAD), f'Missing upload zip: {UPLOAD}'
for p in [INROOT, SRC, DATA, ENG, ONNX, ASSETS]: os.makedirs(p, exist_ok=True)

with zipfile.ZipFile(UPLOAD) as zf:
    zf.extractall(INROOT)

def req(name):
    hits = glob.glob(os.path.join(INROOT, '**', name), recursive=True)
    hits = [h for h in hits if os.path.isfile(h)]
    assert hits, f'Missing inside upload zip: {name}'
    return sorted(hits, key=len)[0]

CHECKPOINT = req('checkpoint (2).pth')
TRAFFIC_PT = req('best (6).pt')
VALZIP = req('valset12.zip')
BUNDLE = req('FINAL_3MODEL_REPRO_BUNDLE.zip')
DEPTH_PT = req('depth_anything_v2_metric_vkitti_vits.pth')

# Restore final bundle snapshots to canonical project paths
RESTORE = os.path.join(PROJECT, 'restored_bundle')
if os.path.isdir(RESTORE): shutil.rmtree(RESTORE)
os.makedirs(RESTORE, exist_ok=True)
with zipfile.ZipFile(BUNDLE) as zf:
    zf.extractall(RESTORE)

def find_dir(root, name, must_contain=None):
    for p in glob.glob(os.path.join(root, '**', name), recursive=True):
        if os.path.isdir(p) and (not must_contain or os.path.exists(os.path.join(p, must_contain))):
            return p
    raise FileNotFoundError(f'Could not find dir {name} under {root}')

def find_file(root, patterns):
    hits=[]
    for pat in patterns:
        hits += glob.glob(os.path.join(root, '**', pat), recursive=True)
    hits=[h for h in hits if os.path.isfile(h)]
    assert hits, f'Missing one of {patterns} under {root}'
    return sorted(hits, key=lambda p: (len(p), p))[0]

YOLOPX_SRC = find_dir(RESTORE, 'YOLOPX', 'lib')
METRIC_SRC = find_dir(RESTORE, 'metric_depth', 'depth_anything_v2')
YOLOPX_DST = os.path.join(SRC, 'YOLOPX')
METRIC_DST = os.path.join(SRC, 'Depth-Anything-V2', 'metric_depth')
if os.path.isdir(YOLOPX_DST): shutil.rmtree(YOLOPX_DST)
if os.path.isdir(os.path.dirname(METRIC_DST)): shutil.rmtree(os.path.dirname(METRIC_DST))
os.makedirs(os.path.dirname(METRIC_DST), exist_ok=True)
shutil.copytree(YOLOPX_SRC, YOLOPX_DST)
shutil.copytree(METRIC_SRC, METRIC_DST)

# Put source weights/checkpoints in fixed locations
shutil.copy2(CHECKPOINT, os.path.join(ASSETS, 'checkpoint (2).pth'))
shutil.copy2(TRAFFIC_PT, os.path.join(ASSETS, 'best (6).pt'))
os.makedirs(os.path.join(METRIC_DST, 'checkpoints'), exist_ok=True)
shutil.copy2(DEPTH_PT, os.path.join(METRIC_DST, 'checkpoints', 'depth_anything_v2_metric_vkitti_vits.pth'))

# Optional ONNXs if included
for pat, outname in [
    ('yolopx_fp32_384x640.onnx','yolopx_fp32_384x640.onnx'),
    ('depth_anything_v2_metric_vkitti_vits_518.onnx','depth_anything_v2_metric_vkitti_vits_518.onnx'),
]:
    hits = glob.glob(os.path.join(INROOT, '**', pat), recursive=True)
    if hits: shutil.copy2(hits[0], os.path.join(ONNX, outname))

# Video from upload or bundle
video = None
for root in [INROOT, RESTORE]:
    hits = glob.glob(os.path.join(root, '**', 'vid10min.mp4'), recursive=True)
    if hits:
        video = hits[0]; break
if video:
    shutil.copy2(video, os.path.join(ASSETS, 'vid10min.mp4'))

# Valset extract
VALROOT = os.path.join(DATA, 'valset')
if os.path.isdir(VALROOT): shutil.rmtree(VALROOT)
os.makedirs(VALROOT, exist_ok=True)
with zipfile.ZipFile(VALZIP) as zf:
    zf.extractall(VALROOT)

# Find the exact valset layout used by final YOLOPX INT8 notebook
candidates = glob.glob(os.path.join(VALROOT, '**', 'imagess', 'val'), recursive=True)
assert candidates, 'Could not find valset folder imagess/val after extracting valset12.zip'
VAL_BASE = os.path.dirname(os.path.dirname(candidates[0]))
needed = ['imagess/val','labelss/val','damaskss/val','lanemasks/val']
for n in needed:
    assert os.path.isdir(os.path.join(VAL_BASE, n)), f'Missing valset subfolder: {os.path.join(VAL_BASE,n)}'

manifest = {
    'project': PROJECT,
    'yolopx_root': YOLOPX_DST,
    'metric_root': METRIC_DST,
    'checkpoint': os.path.join(ASSETS, 'checkpoint (2).pth'),
    'traffic_pt': os.path.join(ASSETS, 'best (6).pt'),
    'depth_pt': os.path.join(METRIC_DST, 'checkpoints', 'depth_anything_v2_metric_vkitti_vits.pth'),
    'val_base': VAL_BASE,
    'video': os.path.join(ASSETS, 'vid10min.mp4') if video else None,
}
with open(os.path.join(PROJECT, 'runpod_paths.json'), 'w') as f: json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2))
print('UNPACK AND VERIFY OK')
