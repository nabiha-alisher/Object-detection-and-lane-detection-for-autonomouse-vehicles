import os, tensorrt as trt, json, shutil
PROJECT=os.environ.get('PROJECT','/workspace/fyp')
ENG_DIR=os.path.join(PROJECT,'engines')
REPO=os.path.join(PROJECT,'model_repo')
os.makedirs(REPO,exist_ok=True)
TRT_LOGGER=trt.Logger(trt.Logger.WARNING)

def dtype_pb(dt):
    return {trt.DataType.FLOAT:'TYPE_FP32',trt.DataType.HALF:'TYPE_FP16',trt.DataType.INT8:'TYPE_INT8',trt.DataType.INT32:'TYPE_INT32'}[dt]

def make_model(model_name, engine_file):
    path=os.path.join(ENG_DIR,engine_file); assert os.path.exists(path), path
    with trt.Runtime(TRT_LOGGER) as rt:
        eng=rt.deserialize_cuda_engine(open(path,'rb').read())
    assert eng is not None, path
    ctx=eng.create_execution_context()
    model_dir=os.path.join(REPO,model_name); ver_dir=os.path.join(model_dir,'1')
    if os.path.isdir(model_dir): shutil.rmtree(model_dir)
    os.makedirs(ver_dir,exist_ok=True)
    shutil.copy2(path,os.path.join(ver_dir,'model.plan'))
    ins=[]; outs=[]
    for i in range(eng.num_io_tensors):
        n=eng.get_tensor_name(i); mode=eng.get_tensor_mode(n); shape=list(ctx.get_tensor_shape(n)); dt=eng.get_tensor_dtype(n)
        dims=shape[1:] if shape and shape[0]==1 else shape
        block=f'  {{\n    name: "{n}"\n    data_type: {dtype_pb(dt)}\n    dims: [ {", ".join(map(str,dims))} ]\n  }}'
        if mode==trt.TensorIOMode.INPUT: ins.append(block)
        else: outs.append(block)
    cfg=f'''name: "{model_name}"
platform: "tensorrt_plan"
max_batch_size: 0

input [
{',\n'.join(ins)}
]

output [
{',\n'.join(outs)}
]
instance_group [ {{ kind: KIND_GPU count: 1 }} ]
'''
    open(os.path.join(model_dir,'config.pbtxt'),'w').write(cfg)
    print(model_name, 'OK')

make_model('yolopx','yolopx_int8_384x640.engine')
make_model('depth_metric','depth_anything_v2_metric_vkitti_vits_fp16.engine')
make_model('traffic','traffic.engine')
print('Triton repo:',REPO)
