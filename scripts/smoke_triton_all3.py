import os
import numpy as np
import tensorrt as trt
import tritonclient.http as httpclient

ENGINE_DIR = "/workspace/fyp/engines"

MODELS = {
    "yolopx": os.path.join(ENGINE_DIR, "yolopx_int8_384x640.engine"),
    "traffic": os.path.join(ENGINE_DIR, "traffic.engine"),
    "depth": os.path.join(ENGINE_DIR, "depth_anything_v2_metric_vkitti_vits_fp16.engine"),
}

def triton_np_dtype(dtype):
    s = str(dtype)
    if "FLOAT" in s:
        return np.float32, "FP32"
    if "HALF" in s:
        return np.float16, "FP16"
    if "INT32" in s:
        return np.int32, "INT32"
    raise RuntimeError(f"Unsupported dtype: {dtype}")

def concrete_shape(shape):
    shape = list(shape)
    # fallback for dynamic dims, mostly not expected here
    for i, d in enumerate(shape):
        if d < 0:
            if len(shape) == 4 and i == 0:
                shape[i] = 1
            elif len(shape) == 4 and i == 1:
                shape[i] = 3
            elif len(shape) == 4:
                shape[i] = 518
            else:
                shape[i] = 1
    return tuple(int(x) for x in shape)

def engine_io(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    assert engine is not None, engine_path

    inputs = []
    outputs = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = concrete_shape(tuple(engine.get_tensor_shape(name)))
        dtype = engine.get_tensor_dtype(name)

        if mode == trt.TensorIOMode.INPUT:
            inputs.append((name, shape, dtype))
        else:
            outputs.append(name)

    return inputs, outputs

client = httpclient.InferenceServerClient(url="localhost:8000")

assert client.is_server_live(), "Triton server not live"
assert client.is_server_ready(), "Triton server not ready"

for model_name, engine_path in MODELS.items():
    print(f"\n=== TESTING {model_name} ===")
    assert client.is_model_ready(model_name), f"{model_name} not ready"

    inputs_meta, output_names = engine_io(engine_path)

    infer_inputs = []
    for name, shape, trt_dtype in inputs_meta:
        np_dtype, triton_dtype = triton_np_dtype(trt_dtype)
        arr = np.random.random(shape).astype(np_dtype)
        inp = httpclient.InferInput(name, shape, triton_dtype)
        inp.set_data_from_numpy(arr)
        infer_inputs.append(inp)
        print("INPUT", name, shape, triton_dtype)

    infer_outputs = [httpclient.InferRequestedOutput(x) for x in output_names]

    result = client.infer(
        model_name=model_name,
        inputs=infer_inputs,
        outputs=infer_outputs,
    )

    for out_name in output_names:
        arr = result.as_numpy(out_name)
        print("OUTPUT", out_name, arr.shape, arr.dtype)

print("\nALL 3 TRITON MODELS INFER OK")
