import os
import sys
import csv
import json
import time
import types
import argparse
import importlib
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tensorrt as trt
import pycuda.driver as cuda
from ultralytics import YOLO
import tritonclient.http as httpclient


# ============================================================
# PATHS — RUNPOD VERSION
# ============================================================
PROJECT = Path("/workspace/fyp")

VIDEO_IN = str(PROJECT / "assets/vid10min.mp4")

YOLOPX_ENGINE = str(PROJECT / "engines/yolopx_int8_384x640.engine")
DEPTH_ENGINE = str(PROJECT / "engines/depth_anything_v2_metric_vkitti_vits_fp16.engine")
TRAFFIC_ENGINE = str(PROJECT / "engines/traffic.engine")

YOLOPX_ROOT = str(PROJECT / "src/YOLOPX")
METRIC_ROOT = str(PROJECT / "src/Depth-Anything-V2/metric_depth")

OUT_DIR = PROJECT / "output/fps_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRITON_URL = "localhost:8000"

YOLOPX_IMGSZ = 640
YOLOPX_CONF_THRES = 0.60
YOLOPX_IOU_THRES = 0.35

TRAFFIC_IMGSZ = 640
TRAFFIC_CONF_THRES = 0.25

DEPTH_INPUT_SIZE = 518

REPORT_EVERY = 100


# ============================================================
# BASIC CHECKS
# ============================================================
for p in [VIDEO_IN, YOLOPX_ENGINE, DEPTH_ENGINE, TRAFFIC_ENGINE]:
    assert os.path.exists(p), f"Missing required file: {p}"

assert os.path.isdir(os.path.join(YOLOPX_ROOT, "lib")), f"Missing YOLOPX lib: {YOLOPX_ROOT}/lib"

cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(0)
except Exception:
    pass

assert torch.cuda.is_available(), "CUDA is not available"
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

print("GPU      :", torch.cuda.get_device_name(0))
print("Torch    :", torch.__version__)
print("CUDA     :", torch.version.cuda)
print("TensorRT :", trt.__version__)
print("OpenCV   :", cv2.__version__)
print("Video    :", VIDEO_IN)
print("YOLOPX   :", YOLOPX_ENGINE)
print("Depth    :", DEPTH_ENGINE)
print("Traffic  :", TRAFFIC_ENGINE)


# ============================================================
# YOLOPX IMPORTS — SAME STYLE AS COLAB
# ============================================================
while YOLOPX_ROOT in sys.path:
    sys.path.remove(YOLOPX_ROOT)
sys.path.insert(0, YOLOPX_ROOT)

for k in list(sys.modules.keys()):
    if k == "lib" or k.startswith("lib."):
        del sys.modules[k]

lib_pkg = types.ModuleType("lib")
lib_pkg.__path__ = [os.path.join(YOLOPX_ROOT, "lib")]
sys.modules["lib"] = lib_pkg
importlib.invalidate_caches()

from lib.core.general import non_max_suppression, scale_coords
from lib.utils import letterbox_for_img

print("YOLOPX imports OK")


# ============================================================
# GPU NORMALIZATION TENSORS
# ============================================================
_YOLOPX_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
_YOLOPX_STD = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)

_DEPTH_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
_DEPTH_STD = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)


def compute_depth_target_size(h, w, input_size=518):
    scale = input_size / min(h, w)
    new_h = int(round(h * scale / 14.0)) * 14
    new_w = int(round(w * scale / 14.0)) * 14
    new_h = max(new_h, 14)
    new_w = max(new_w, 14)
    return int(new_h), int(new_w)


DEPTH_TARGET_H = None
DEPTH_TARGET_W = None


# ============================================================
# PREPROCESS FUNCTIONS
# ============================================================
def preprocess_yolopx_fast(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = frame_rgb.shape[:2]

    input_img, ratio, pad = letterbox_for_img(frame_rgb, YOLOPX_IMGSZ, auto=True)
    h, w = input_img.shape[:2]
    shapes = (h0, w0), ((h / h0, w / w0), pad)

    arr = np.ascontiguousarray(input_img.transpose(2, 0, 1), dtype=np.float32)
    arr /= 255.0

    x = torch.from_numpy(arr).unsqueeze(0).to(device, non_blocking=True)
    x.sub_(_YOLOPX_MEAN).div_(_YOLOPX_STD)
    return x, shapes


def preprocess_depth_fast(frame_bgr):
    global DEPTH_TARGET_H, DEPTH_TARGET_W
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_res = cv2.resize(
        img_rgb,
        (DEPTH_TARGET_W, DEPTH_TARGET_H),
        interpolation=cv2.INTER_CUBIC,
    )
    arr = np.ascontiguousarray(img_res.transpose(2, 0, 1), dtype=np.float32)
    x = torch.from_numpy(arr).unsqueeze(0).to(device, non_blocking=True)
    x.sub_(_DEPTH_MEAN).div_(_DEPTH_STD)
    return x


def letterbox_np(im, new_shape=640, color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im


def preprocess_traffic_for_triton(frame_bgr):
    img = letterbox_np(frame_bgr, TRAFFIC_IMGSZ)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return arr[None, ...]


# ============================================================
# RAW TENSORRT RUNNER — SAME STYLE AS COLAB
# ============================================================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def trt_dtype_to_torch(dt):
    if dt == trt.DataType.FLOAT:
        return torch.float32
    if dt == trt.DataType.HALF:
        return torch.float16
    if dt == trt.DataType.INT8:
        return torch.int8
    if dt == trt.DataType.INT32:
        return torch.int32
    raise RuntimeError(f"Unsupported TRT dtype: {dt}")


class RawTRTRunner:
    def __init__(self, engine_path, name):
        self.name = name
        self.engine_path = engine_path

        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create execution context: {engine_path}")

        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]

        self.input_name = None
        self.output_names = []
        for n in self.tensor_names:
            mode = self.engine.get_tensor_mode(n)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = n
            else:
                self.output_names.append(n)

        assert self.input_name is not None, engine_path

        self.input_shape = tuple(self.context.get_tensor_shape(self.input_name))
        try:
            self.context.set_input_shape(self.input_name, self.input_shape)
        except Exception:
            pass

        self.input_dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(self.input_name))
        self.input_buf = torch.empty(self.input_shape, device=device, dtype=self.input_dtype)

        self.output_tensors = {}
        for n in self.output_names:
            shp = tuple(self.context.get_tensor_shape(n))
            dt = trt_dtype_to_torch(self.engine.get_tensor_dtype(n))
            t = torch.empty(shp, device=device, dtype=dt)
            self.output_tensors[n] = t
            self.context.set_tensor_address(n, int(t.data_ptr()))

        self.e0 = torch.cuda.Event(enable_timing=True)
        self.e1 = torch.cuda.Event(enable_timing=True)

        print(f"\nRAW TRT {self.name}")
        print(" input :", self.input_name, self.input_shape, self.input_dtype)
        for n in self.output_names:
            print(" output:", n, tuple(self.context.get_tensor_shape(n)), self.output_tensors[n].dtype)

    def warmup(self, x, iters=10):
        for _ in range(iters):
            self.forward(x)
        torch.cuda.synchronize()

    def forward(self, x):
        assert tuple(x.shape) == tuple(self.input_shape), (self.name, tuple(x.shape), tuple(self.input_shape))

        if self.input_buf.dtype == torch.float16:
            self.input_buf.copy_(x.half())
        else:
            self.input_buf.copy_(x.float())

        self.context.set_tensor_address(self.input_name, int(self.input_buf.data_ptr()))

        stream = torch.cuda.current_stream()

        self.e0.record()
        ok = self.context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError(f"{self.name} TRT execute_async_v3 failed")
        self.e1.record()

        torch.cuda.synchronize()
        ms = self.e0.elapsed_time(self.e1)

        outs = {k: v.float() for k, v in self.output_tensors.items()}
        return outs, ms


# ============================================================
# TRITON HELPERS
# ============================================================
class TritonRunner:
    def __init__(self, client, model_name, input_name, output_names):
        self.client = client
        self.model_name = model_name
        self.input_name = input_name
        self.output_names = output_names

    def infer_np(self, arr):
        inp = httpclient.InferInput(self.input_name, arr.shape, "FP32")
        inp.set_data_from_numpy(arr.astype(np.float32, copy=False))
        outs = [httpclient.InferRequestedOutput(n) for n in self.output_names]

        t0 = time.perf_counter()
        res = self.client.infer(self.model_name, inputs=[inp], outputs=outs)
        total_ms = (time.perf_counter() - t0) * 1000.0

        out = {n: res.as_numpy(n) for n in self.output_names}
        return out, total_ms


# ============================================================
# POSTPROCESS HELPERS
# ============================================================
def yolopx_lane_mask(x, shapes, ll_seg_out, out_h, out_w):
    _, _, H, W = x.shape
    pad_w, pad_h = shapes[1][1]
    pad_w = int(pad_w)
    pad_h = int(pad_h)

    ll_predict = ll_seg_out[:, :, pad_h:(H - pad_h), pad_w:(W - pad_w)]
    ll_seg_mask = F.interpolate(ll_predict, size=(out_h, out_w), mode="bilinear")
    _, ll_seg_mask = torch.max(ll_seg_mask, 1)
    return ll_seg_mask.int().squeeze().detach().cpu().numpy()


def summarize_ms(rows, keys):
    out = {}
    n = max(len(rows), 1)
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if not vals:
            out[k] = {"avg_ms": None, "fps": None}
            continue
        avg = float(sum(vals) / len(vals))
        out[k] = {
            "avg_ms": avg,
            "fps": float(1000.0 / avg) if avg > 0 else None,
        }
    return out


def print_block(title, data):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    for k, v in data.items():
        if v["avg_ms"] is None:
            continue
        print(f"{k:30s} {v['avg_ms']:9.3f} ms   {v['fps']:9.3f} FPS")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--mode", choices=["both", "triton", "raw"], default="both")
    parser.add_argument("--report-every", type=int, default=100)
    args = parser.parse_args()

    global DEPTH_TARGET_H, DEPTH_TARGET_W
    REPORT_EVERY_LOCAL = args.report_every

    cap = cv2.VideoCapture(VIDEO_IN)
    assert cap.isOpened(), VIDEO_IN

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    DEPTH_TARGET_H, DEPTH_TARGET_W = compute_depth_target_size(src_h, src_w, DEPTH_INPUT_SIZE)

    print("\nVIDEO META")
    print("fps          :", src_fps)
    print("total_frames :", total_frames)
    print("resolution   :", f"{src_w}x{src_h}")
    print("profile start:", args.start)
    print("profile n    :", args.frames)
    print("depth target :", f"{DEPTH_TARGET_W}x{DEPTH_TARGET_H}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    ok, warm_frame = cap.read()
    assert ok, "Could not read warmup frame"
    cap.release()

    yx, y_shapes = preprocess_yolopx_fast(warm_frame)
    dx = preprocess_depth_fast(warm_frame)
    tx = preprocess_traffic_for_triton(warm_frame)

    raw_yolo = raw_depth = raw_traffic = None
    traffic_model = None

    if args.mode in ["both", "raw"]:
        print("\nInitializing RAW TensorRT runners...")
        cuda.init()
        ctx = cuda.Device(0).retain_primary_context()
        ctx.push()

        raw_yolo = RawTRTRunner(YOLOPX_ENGINE, "YOLOPX")
        raw_depth = RawTRTRunner(DEPTH_ENGINE, "DEPTH")

        print("\nInitializing RAW Ultralytics traffic engine...")
        traffic_model = YOLO(TRAFFIC_ENGINE, task="detect")

        raw_yolo.warmup(yx, iters=10)
        raw_depth.warmup(dx, iters=10)
        _ = traffic_model.predict(
            source=warm_frame,
            imgsz=TRAFFIC_IMGSZ,
            conf=TRAFFIC_CONF_THRES,
            device=0,
            verbose=False,
        )

    triton_yolo = triton_depth = triton_traffic = None
    triton_client = None

    if args.mode in ["both", "triton"]:
        print("\nInitializing Triton HTTP client...")
        triton_client = httpclient.InferenceServerClient(url=TRITON_URL)

        assert triton_client.is_server_live(), "Triton server not live"
        assert triton_client.is_server_ready(), "Triton server not ready"

        triton_yolo = TritonRunner(
            triton_client,
            "yolopx",
            "images",
            ["det_out", "da_seg", "ll_seg"],
        )
        triton_depth = TritonRunner(
            triton_client,
            "depth",
            "images",
            ["depth"],
        )
        triton_traffic = TritonRunner(
            triton_client,
            "traffic",
            "images",
            ["output0"],
        )

        _ = triton_yolo.infer_np(yx.detach().cpu().numpy())
        _ = triton_depth.infer_np(dx.detach().cpu().numpy())
        _ = triton_traffic.infer_np(tx)

    cap = cv2.VideoCapture(VIDEO_IN)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    rows = []
    raw_rows = []
    triton_rows = []

    print("\nRUNNING FPS PROFILER...")

    t_all0 = time.perf_counter()

    for i in range(args.frames):
        row = {"profile_frame": i, "video_frame": args.start + i}

        t0 = time.perf_counter()
        ok, frame = cap.read()
        row["read_ms"] = (time.perf_counter() - t0) * 1000.0
        if not ok:
            break

        # -------------------------
        # shared preprocessing
        # -------------------------
        t0 = time.perf_counter()
        yx, y_shapes = preprocess_yolopx_fast(frame)
        row["prep_yolopx_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        dx = preprocess_depth_fast(frame)
        row["prep_depth_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        tx = preprocess_traffic_for_triton(frame)
        row["prep_traffic_ms"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------
        # RAW TRT path
        # -------------------------
        if args.mode in ["both", "raw"]:
            t_raw0 = time.perf_counter()

            y_out_raw, y_ms_raw = raw_yolo.forward(yx)
            d_out_raw, d_ms_raw = raw_depth.forward(dx)

            t0 = time.perf_counter()
            traffic_res = traffic_model.predict(
                source=frame,
                imgsz=TRAFFIC_IMGSZ,
                conf=TRAFFIC_CONF_THRES,
                device=0,
                verbose=False,
            )[0]
            raw_traffic_total_ms = (time.perf_counter() - t0) * 1000.0
            raw_traffic_inf_ms = float(traffic_res.speed["inference"])

            t0 = time.perf_counter()
            det_pred = non_max_suppression(
                y_out_raw["det_out"],
                conf_thres=YOLOPX_CONF_THRES,
                iou_thres=YOLOPX_IOU_THRES,
                classes=None,
                agnostic=False,
            )
            det = det_pred[0]
            raw_nms_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            lane_mask = yolopx_lane_mask(yx, y_shapes, y_out_raw["ll_seg"], src_h, src_w)
            raw_lane_mask_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            depth_map = next(iter(d_out_raw.values())).squeeze().cpu().numpy()
            raw_depth_cpu_ms = (time.perf_counter() - t0) * 1000.0

            raw_total_ms = (time.perf_counter() - t_raw0) * 1000.0

            raw_row = {
                "profile_frame": i,
                "raw_yolopx_forward_ms": y_ms_raw,
                "raw_depth_forward_ms": d_ms_raw,
                "raw_traffic_total_ms": raw_traffic_total_ms,
                "raw_traffic_inference_ms": raw_traffic_inf_ms,
                "raw_yolopx_nms_ms": raw_nms_ms,
                "raw_lane_mask_ms": raw_lane_mask_ms,
                "raw_depth_cpu_ms": raw_depth_cpu_ms,
                "raw_backend_total_ms": raw_total_ms,
            }
            raw_rows.append(raw_row)
            row.update(raw_row)

        # -------------------------
        # Triton HTTP path
        # -------------------------
        if args.mode in ["both", "triton"]:
            t_tri0 = time.perf_counter()

            y_np = yx.detach().cpu().numpy()
            d_np = dx.detach().cpu().numpy()

            y_out_tri_np, y_tri_ms = triton_yolo.infer_np(y_np)
            d_out_tri_np, d_tri_ms = triton_depth.infer_np(d_np)
            t_out_tri_np, t_tri_ms = triton_traffic.infer_np(tx)

            t0 = time.perf_counter()
            y_out_tri = {
                "det_out": torch.from_numpy(y_out_tri_np["det_out"]).to(device),
                "da_seg": torch.from_numpy(y_out_tri_np["da_seg"]).to(device),
                "ll_seg": torch.from_numpy(y_out_tri_np["ll_seg"]).to(device),
            }
            tri_to_torch_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            det_pred = non_max_suppression(
                y_out_tri["det_out"],
                conf_thres=YOLOPX_CONF_THRES,
                iou_thres=YOLOPX_IOU_THRES,
                classes=None,
                agnostic=False,
            )
            det = det_pred[0]
            tri_nms_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            lane_mask = yolopx_lane_mask(yx, y_shapes, y_out_tri["ll_seg"], src_h, src_w)
            tri_lane_mask_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            depth_map = d_out_tri_np["depth"].squeeze()
            tri_depth_cpu_ms = (time.perf_counter() - t0) * 1000.0

            tri_total_ms = (time.perf_counter() - t_tri0) * 1000.0

            tri_row = {
                "profile_frame": i,
                "triton_yolopx_http_ms": y_tri_ms,
                "triton_depth_http_ms": d_tri_ms,
                "triton_traffic_http_ms": t_tri_ms,
                "triton_outputs_to_torch_ms": tri_to_torch_ms,
                "triton_yolopx_nms_ms": tri_nms_ms,
                "triton_lane_mask_ms": tri_lane_mask_ms,
                "triton_depth_cpu_ms": tri_depth_cpu_ms,
                "triton_backend_total_ms": tri_total_ms,
            }
            triton_rows.append(tri_row)
            row.update(tri_row)

        rows.append(row)

        done = i + 1
        if done % REPORT_EVERY_LOCAL == 0:
            elapsed = time.perf_counter() - t_all0
            msg = f"{done:5d}/{args.frames} | wall FPS={done / max(elapsed, 1e-9):7.3f}"

            if raw_rows:
                avg_raw = sum(r["raw_backend_total_ms"] for r in raw_rows) / len(raw_rows)
                msg += f" | RAW backend FPS={1000.0 / avg_raw:7.3f}"

            if triton_rows:
                avg_tri = sum(r["triton_backend_total_ms"] for r in triton_rows) / len(triton_rows)
                msg += f" | Triton backend FPS={1000.0 / avg_tri:7.3f}"

            print(msg)

    cap.release()

    # ============================================================
    # WRITE CSV + JSON
    # ============================================================
    tag = f"{args.mode}_{len(rows)}frames"
    csv_path = OUT_DIR / f"fps_compare_{tag}.csv"
    json_path = OUT_DIR / f"fps_compare_{tag}_summary.json"

    all_keys = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)

    common_keys = [
        "read_ms",
        "prep_yolopx_ms",
        "prep_depth_ms",
        "prep_traffic_ms",
    ]

    raw_keys = [
        "raw_yolopx_forward_ms",
        "raw_depth_forward_ms",
        "raw_traffic_total_ms",
        "raw_traffic_inference_ms",
        "raw_yolopx_nms_ms",
        "raw_lane_mask_ms",
        "raw_depth_cpu_ms",
        "raw_backend_total_ms",
    ]

    triton_keys = [
        "triton_yolopx_http_ms",
        "triton_depth_http_ms",
        "triton_traffic_http_ms",
        "triton_outputs_to_torch_ms",
        "triton_yolopx_nms_ms",
        "triton_lane_mask_ms",
        "triton_depth_cpu_ms",
        "triton_backend_total_ms",
    ]

    summary = {
        "mode": args.mode,
        "frames_requested": args.frames,
        "frames_profiled": len(rows),
        "video": VIDEO_IN,
        "resolution": f"{src_w}x{src_h}",
        "source_fps": src_fps,
        "common": summarize_ms(rows, common_keys),
        "raw": summarize_ms(raw_rows, raw_keys) if raw_rows else {},
        "triton": summarize_ms(triton_rows, triton_keys) if triton_rows else {},
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
        },
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_block("COMMON CPU/GPU PREPROCESS TIMINGS", summary["common"])

    if raw_rows:
        print_block("RAW LOCAL TENSORRT TIMINGS", summary["raw"])

    if triton_rows:
        print_block("TRITON HTTP TIMINGS", summary["triton"])

    if raw_rows and triton_rows:
        raw_total = summary["raw"]["raw_backend_total_ms"]["avg_ms"]
        tri_total = summary["triton"]["triton_backend_total_ms"]["avg_ms"]

        print("\n" + "=" * 70)
        print("DIRECT COMPARISON")
        print("=" * 70)
        print(f"RAW backend total      : {raw_total:.3f} ms/frame | {1000.0/raw_total:.3f} FPS")
        print(f"Triton backend total   : {tri_total:.3f} ms/frame | {1000.0/tri_total:.3f} FPS")
        print(f"Triton minus RAW       : {tri_total - raw_total:.3f} ms/frame")
        print(f"Triton/RAW slowdown    : {tri_total / raw_total:.3f}x")

    print("\nSaved CSV :", csv_path)
    print("Saved JSON:", json_path)


if __name__ == "__main__":
    main()
