import os
import shutil
import tensorrt as trt
from pathlib import Path

PROJECT = Path("/workspace/fyp")
ENGINE_DIR = PROJECT / "engines"
REPO = PROJECT / "triton_model_repo"

MODELS = {
    "yolopx": ENGINE_DIR / "yolopx_int8_384x640.engine",
    "traffic": ENGINE_DIR / "traffic.engine",
    "depth": ENGINE_DIR / "depth_anything_v2_metric_vkitti_vits_fp16.engine",
}

def triton_dtype(dtype):
    s = str(dtype)
    if "FLOAT" in s:
        return "TYPE_FP32"
    if "HALF" in s:
        return "TYPE_FP16"
    if "INT32" in s:
        return "TYPE_INT32"
    if "INT64" in s:
        return "TYPE_INT64"
    if "BOOL" in s:
        return "TYPE_BOOL"
    raise RuntimeError(f"Unsupported TensorRT dtype: {dtype}")

def dims_txt(shape):
    return "[ " + ", ".join(str(int(x)) for x in shape) + " ]"

def make_one(model_name, engine_path):
    assert engine_path.exists(), f"Missing engine: {engine_path}"

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())

    assert engine is not None, f"Could not deserialize: {engine_path}"

    model_dir = REPO / model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(engine_path, version_dir / "model.plan")

    inputs = []
    outputs = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = triton_dtype(engine.get_tensor_dtype(name))

        item = {
            "name": name,
            "shape": shape,
            "dtype": dtype,
        }

        if mode == trt.TensorIOMode.INPUT:
            inputs.append(item)
        else:
            outputs.append(item)

    assert inputs, f"No inputs found for {model_name}"
    assert outputs, f"No outputs found for {model_name}"

    config = []
    config.append(f'name: "{model_name}"')
    config.append('backend: "tensorrt"')
    config.append("max_batch_size: 0")
    config.append("")
    config.append("input [")
    for j, x in enumerate(inputs):
        comma = "," if j < len(inputs) - 1 else ""
        config.append("  {")
        config.append(f'    name: "{x["name"]}"')
        config.append(f'    data_type: {x["dtype"]}')
        config.append(f'    dims: {dims_txt(x["shape"])}')
        config.append(f"  }}{comma}")
    config.append("]")
    config.append("")
    config.append("output [")
    for j, x in enumerate(outputs):
        comma = "," if j < len(outputs) - 1 else ""
        config.append("  {")
        config.append(f'    name: "{x["name"]}"')
        config.append(f'    data_type: {x["dtype"]}')
        config.append(f'    dims: {dims_txt(x["shape"])}')
        config.append(f"  }}{comma}")
    config.append("]")
    config.append("")

    (model_dir / "config.pbtxt").write_text("\n".join(config))

    print(f"\n=== {model_name} ===")
    print("Engine:", engine_path)
    print("Copied to:", version_dir / "model.plan")
    print("Config:", model_dir / "config.pbtxt")
    for x in inputs:
        print("INPUT ", x["name"], x["shape"], x["dtype"])
    for x in outputs:
        print("OUTPUT", x["name"], x["shape"], x["dtype"])

if REPO.exists():
    shutil.rmtree(REPO)
REPO.mkdir(parents=True, exist_ok=True)

for name, path in MODELS.items():
    make_one(name, path)

print("\nTRITON MODEL REPO READY:", REPO)
