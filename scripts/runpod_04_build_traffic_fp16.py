import os, json, shutil
from ultralytics import YOLO

PROJECT=os.environ.get('PROJECT','/workspace/fyp')
paths=json.load(open(os.path.join(PROJECT,'runpod_paths.json')))
PT=paths['traffic_pt']
ENG_DIR=os.path.join(PROJECT,'engines'); ONNX_DIR=os.path.join(PROJECT,'onnx')
os.makedirs(ENG_DIR,exist_ok=True); os.makedirs(ONNX_DIR,exist_ok=True)
assert os.path.exists(PT), PT
m=YOLO(PT)
print('Traffic class names:', m.names)
need={'tl_red','tl_yellow','tl_green','tl_none'}
have=set(map(str,m.names.values())) if isinstance(m.names,dict) else set(map(str,m.names))
assert need.issubset(have), f'Traffic class mismatch: {have}'
# Ultralytics TensorRT export. This creates .engine beside the .pt by default.
exported = m.export(format='engine', imgsz=640, half=True, dynamic=False, batch=1, simplify=False, device=0)
print('Ultralytics exported:', exported)
if not exported or not os.path.exists(str(exported)):
    # common fallback: same basename .engine
    exported = os.path.splitext(PT)[0] + '.engine'
assert os.path.exists(str(exported)), f'Engine not found after export: {exported}'
out=os.path.join(ENG_DIR,'traffic.engine')
shutil.copy2(str(exported), out)
print('Saved traffic engine:', out)
