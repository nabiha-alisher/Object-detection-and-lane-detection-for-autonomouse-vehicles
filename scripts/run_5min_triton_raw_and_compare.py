import os
import re
import sys
import json
import time
import shutil
import hashlib
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np


PROJECT = Path("/workspace/fyp")
TRITON_PIPELINE = PROJECT / "final_triton_preserved_pipeline.py"
RAW_PIPELINE = PROJECT / "final_rawtrt_preserved_pipeline.py"

OUT_TRITON = PROJECT / "output/COMPARE_TRITON_5MIN"
OUT_RAW = PROJECT / "output/COMPARE_RAWTRT_5MIN"
OUT_REPORT = PROJECT / "output/COMPARE_5MIN_REPORT"

LOG_TRITON = PROJECT / "logs/run_triton_5min.log"
LOG_RAW = PROJECT / "logs/run_rawtrt_5min.log"

REPORT_JSON = OUT_REPORT / "triton_vs_raw_5min_compare_report.json"
REPORT_MD = OUT_REPORT / "triton_vs_raw_5min_compare_report.md"
MISMATCH_CSV = OUT_REPORT / "triton_vs_raw_5min_csv_mismatches_sample.csv"

VIDEO = PROJECT / "assets/vid10min.mp4"
YOLOPX_ENGINE = PROJECT / "engines/yolopx_int8_384x640.engine"
DEPTH_ENGINE = PROJECT / "engines/depth_anything_v2_metric_vkitti_vits_fp16.engine"
TRAFFIC_ENGINE = PROJECT / "engines/traffic.engine"

EXPECTED_FRAMES = 9000


def run(cmd, *, env=None, log_path=None, cwd=PROJECT, check=True):
    print("\n" + "=" * 90)
    print("RUN:", " ".join(map(str, cmd)))
    print("=" * 90)

    if log_path:
        with open(log_path, "w") as log:
            p = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in p.stdout:
                print(line, end="")
                log.write(line)
            rc = p.wait()
    else:
        rc = subprocess.call(cmd, cwd=str(cwd), env=env)

    if check and rc != 0:
        raise SystemExit(f"Command failed with rc={rc}: {' '.join(map(str, cmd))}")

    return rc


def sha256(path: Path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_exists(path: Path, label: str):
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def strip_bottom_autorun(src: str) -> str:
    """
    Remove existing bottom:
        result = run_first_minutes_pipeline(...)
        result
    Then we append our own controlled 5-min call.
    """
    patterns = [
        r'\nresult\s*=\s*run_first_minutes_pipeline\([\s\S]*?\)\s*\nresult\s*\n?\s*$',
        r'\nif\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:\s*[\s\S]*$',
    ]

    out = src
    for pat in patterns:
        out2 = re.sub(pat, "\n", out, flags=re.MULTILINE)
        if out2 != out:
            out = out2
    return out.rstrip() + "\n"


def ensure_import(src: str, import_line: str, after_regex: str):
    if import_line in src:
        return src
    m = re.search(after_regex, src)
    if not m:
        raise SystemExit(f"Could not insert import line: {import_line}")
    return src[:m.end()] + "\n" + import_line + src[m.end():]


def replace_top_level_assignment(src: str, var_name: str, replacement_line: str):
    """
    Replace a top-level assignment, including multi-line constructor calls.

    Example:
      yolopx_runner = TritonRunner(
          ...
      )
    becomes:
      yolopx_runner = RawTRTRunner(...)
    """
    lines = src.splitlines()
    target_i = None

    for i, line in enumerate(lines):
        if line.startswith(f"{var_name} "):
            if re.match(rf"^{re.escape(var_name)}\s*=", line):
                target_i = i
                break
        if line.startswith(f"{var_name}="):
            target_i = i
            break

    if target_i is None:
        raise SystemExit(f"Could not find top-level assignment for {var_name}")

    start = target_i
    bal = 0
    end = start

    for j in range(start, len(lines)):
        line = lines[j]
        stripped = line.strip()

        # Rough but enough for constructor blocks.
        bal += line.count("(") + line.count("[") + line.count("{")
        bal -= line.count(")") + line.count("]") + line.count("}")

        end = j

        if j == start:
            # Single-line assignment, no open parens.
            if bal <= 0 and not stripped.endswith("\\"):
                break
        else:
            if bal <= 0:
                break

    new_lines = lines[:start] + [replacement_line] + lines[end + 1:]
    return "\n".join(new_lines) + "\n"


RAW_RUNNER_CODE = r'''
# ==============================================================
# RAW LOCAL TENSORRT PATCH\nTRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# This replaces only the inference backend.
# Decision logic, draw logic, CSV logic, video writer, thresholds:
# inherited from final_triton_preserved_pipeline.py.
# ==============================================================

try:
    if "_ctx" in globals() and _ctx is not None:
        try:
            _ctx.pop()
        except Exception:
            pass
except Exception:
    pass

cuda.init()
_ctx = cuda.Device(0).retain_primary_context()
_ctx.push()

# RAW_LOCAL_TRT_PATCH_ATEXIT_CTX_POP
def _rawtrt_cleanup_context():
    try:
        if "_ctx" in globals() and _ctx is not None:
            _ctx.pop()
            print("CUDA context popped by atexit.")
    except Exception as e:
        print("CUDA context atexit pop warning:", repr(e))

atexit.register(_rawtrt_cleanup_context)

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
    def __init__(self, engine_path, name="RAW_TRT"):
        self.name = name
        self.engine_path = engine_path

        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create execution context: {engine_path}")

        self.tensor_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
        ]

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

        print(f"RawTRT runner ready: {name}")
        print(" input :", self.input_name, self.input_shape, self.input_dtype)
        print(" output:", self.output_names)

    def warmup(self, x, iters=10):
        assert tuple(x.shape) == tuple(self.input_shape), (tuple(x.shape), tuple(self.input_shape))
        for _ in range(iters):
            if self.input_buf.dtype == torch.float16:
                self.input_buf.copy_(x.half())
            else:
                self.input_buf.copy_(x.float())

            self.context.set_tensor_address(self.input_name, int(self.input_buf.data_ptr()))
            stream = torch.cuda.current_stream()
            ok = self.context.execute_async_v3(stream.cuda_stream)
            if not ok:
                raise RuntimeError(f"{self.name} TRT warmup failed")

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
'''


def create_raw_pipeline():
    assert_exists(TRITON_PIPELINE, "existing Triton pipeline")

    src = TRITON_PIPELINE.read_text()

    # Keep same codebase but strip autorun so we control the call.
    src = strip_bottom_autorun(src)

    # Ensure pycuda import.
    src = ensure_import(
        src,
        "import pycuda.driver as cuda",
        r"import tensorrt as trt"
    )

    # Remove Triton hard requirement. Leave unused Triton class/import harmless.
    src = re.sub(r'^TRITON_URL\s*=.*$', 'TRITON_URL = None  # RAW_LOCAL_TRT_PATCH', src, flags=re.MULTILINE)
    src = re.sub(r'^triton_client\s*=.*$', 'triton_client = None  # RAW_LOCAL_TRT_PATCH', src, flags=re.MULTILINE)
    src = re.sub(r'^assert\s+triton_client\..*$', '# RAW_LOCAL_TRT_PATCH: Triton assert removed', src, flags=re.MULTILINE)
    src = re.sub(r'^print\("Triton URL:".*$', 'print("Inference backend: RAW_LOCAL_TENSORRT")', src, flags=re.MULTILINE)

    # Insert RawTRTRunner before first runner assignment.
    if "class RawTRTRunner:" not in src:
        m = re.search(r'^yolopx_runner\s*=', src, flags=re.MULTILINE)
        if not m:
            raise SystemExit("Could not find yolopx_runner assignment insertion point")
        src = src[:m.start()] + RAW_RUNNER_CODE + "\n" + src[m.start():]

    # Replace runner assignments. This is the actual backend switch.
    src = replace_top_level_assignment(
        src,
        "yolopx_runner",
        'yolopx_runner = RawTRTRunner(YOLOPX_ENGINE, "YOLOPX_RAW")'
    )
    src = replace_top_level_assignment(
        src,
        "depth_runner",
        'depth_runner = RawTRTRunner(DEPTH_ENGINE, "DEPTH_RAW")'
    )

    # RAW_LOCAL_TRT_PATCH:
    # Traffic must also be local, not TrafficTritonModel.
    # This keeps the traffic engine path same, only backend changes.
    src = replace_top_level_assignment(
        src,
        "traffic_model",
        'traffic_model = YOLO(TRAFFIC_ENGINE, task="detect")'
    )

    # Force a clear raw output label in notes if existing string exists.
    src = src.replace(
        '"speed_build":          "parallel-preprocess + band-cache + async-write + LUT-overlay + GPU-normalize"',
        '"speed_build":          "RAW_LOCAL_TENSORRT + parallel-preprocess + band-cache + async-write + LUT-overlay + GPU-normalize"'
    )

    # Controlled bottom call: same first 5 min, same suffix, same lane strategy.
    src += r'''

if __name__ == "__main__":
    try:
        result = run_first_minutes_pipeline(
            minutes=5.0,
            preview_only=False,
            name_suffix="speed_optimized_first5min",
            lane_strategy="corridor_memory_two_hits",
        )
        print("\nRAW_LOCAL_TENSORRT_RUN_RESULT_JSON")
        print(json.dumps(result, indent=2))
    finally:
        if "_ctx" in globals() and _ctx is not None:
            try:
                _ctx.pop()
                print("CUDA context popped cleanly.")
            except Exception as e:
                print("CUDA context pop warning:", repr(e))
'''

    RAW_PIPELINE.write_text(src)

    # Syntax check.
    run([sys.executable, "-m", "py_compile", str(RAW_PIPELINE)], check=True)

    print("\nCreated raw pipeline:", RAW_PIPELINE)
    print("Raw pipeline sha256 :", sha256(RAW_PIPELINE))


def find_single(pattern: str, folder: Path):
    hits = sorted(folder.glob(pattern))
    if not hits:
        raise SystemExit(f"No file found: {folder}/{pattern}")
    if len(hits) > 1:
        print("Multiple matches; using newest:")
        for h in hits:
            print(" ", h)
        hits = sorted(hits, key=lambda p: p.stat().st_mtime)
    return hits[-1]


def run_pipeline_pair():
    assert_exists(VIDEO, "video")
    assert_exists(YOLOPX_ENGINE, "YOLOPX engine")
    assert_exists(DEPTH_ENGINE, "Depth engine")
    assert_exists(TRAFFIC_ENGINE, "Traffic engine")

    OUT_TRITON.mkdir(parents=True, exist_ok=True)
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.mkdir(parents=True, exist_ok=True)

    # Clean output folders only for this comparison.
    for folder in [OUT_TRITON, OUT_RAW, OUT_REPORT]:
        for p in folder.glob("*"):
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    # Triton health check.
    print("\nChecking Triton readiness...")
    rc = subprocess.call(
        ["bash", "-lc", "curl -s -o /dev/null -w '%{http_code}' localhost:8000/v2/health/ready"],
        cwd=str(PROJECT)
    )
    print()
    # Also direct string check for clear fail.
    code = subprocess.check_output(
        ["bash", "-lc", "curl -s -o /dev/null -w '%{http_code}' localhost:8000/v2/health/ready"],
        cwd=str(PROJECT),
        text=True
    ).strip()
    if code != "200":
        raise SystemExit(f"Triton not ready. HTTP={code}")

    common_env = os.environ.copy()
    common_env["PYTHONUNBUFFERED"] = "1"
    common_env["LIVE_PREVIEW_DIR"] = "/workspace/fyp/output/COMPARE_5MIN_REPORT/live_preview_dummy"

    # Run Triton existing pipeline.
    env_tri = common_env.copy()
    env_tri["OUT_DIR"] = str(OUT_TRITON)
    env_tri["TRITON_URL"] = "localhost:8000"

    print("\n\n############################")
    print("# RUNNING TRITON 5-MIN PIPELINE")
    print("############################")
    run(
        [sys.executable, "-u", str(TRITON_PIPELINE)],
        env=env_tri,
        log_path=LOG_TRITON,
        check=True
    )

    # Run RawTRT generated pipeline.
    env_raw = common_env.copy()
    env_raw["OUT_DIR"] = str(OUT_RAW)

    print("\n\n############################")
    print("# RUNNING RAW LOCAL TRT 5-MIN PIPELINE")
    print("############################")
    run(
        [sys.executable, "-u", str(RAW_PIPELINE)],
        env=env_raw,
        log_path=LOG_RAW,
        check=True
    )


def load_outputs():
    tri_csv = find_single("*_frames.csv", OUT_TRITON)
    raw_csv = find_single("*_frames.csv", OUT_RAW)

    tri_json = find_single("*_summary.json", OUT_TRITON)
    raw_json = find_single("*_summary.json", OUT_RAW)

    tri_mp4 = find_single("*.mp4", OUT_TRITON)
    raw_mp4 = find_single("*.mp4", OUT_RAW)

    return {
        "tri_csv": tri_csv,
        "raw_csv": raw_csv,
        "tri_json": tri_json,
        "raw_json": raw_json,
        "tri_mp4": tri_mp4,
        "raw_mp4": raw_mp4,
    }


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def compare_csvs(tri_csv: Path, raw_csv: Path):
    tri = pd.read_csv(tri_csv)
    raw = pd.read_csv(raw_csv)

    report = {
        "triton_rows": int(len(tri)),
        "raw_rows": int(len(raw)),
        "same_row_count": bool(len(tri) == len(raw)),
        "expected_frames": EXPECTED_FRAMES,
        "triton_expected_frames": bool(len(tri) == EXPECTED_FRAMES),
        "raw_expected_frames": bool(len(raw) == EXPECTED_FRAMES),
    }

    key_cols = [
        "frame_index",
        "time_sec",
        "traffic_stop",
        "traffic_ready",
        "traffic_go",
        "traffic_active_class",
        "traffic_num_boxes",
        "lane_left_switch",
        "lane_right_switch",
        "lane_found",
        "yolopx_box_count",
    ]

    float_cols = [
        "traffic_active_conf",
    ]

    common_key_cols = [c for c in key_cols if c in tri.columns and c in raw.columns]
    common_float_cols = [c for c in float_cols if c in tri.columns and c in raw.columns]

    n = min(len(tri), len(raw))
    mismatches = []

    exact_mismatch_counts = {}
    for c in common_key_cols:
        a = tri[c].iloc[:n].astype(str).fillna("")
        b = raw[c].iloc[:n].astype(str).fillna("")
        mask = a.ne(b)
        count = int(mask.sum())
        exact_mismatch_counts[c] = count

        if count:
            idxs = list(mask[mask].index[:50])
            for idx in idxs:
                mismatches.append({
                    "row": int(idx),
                    "column": c,
                    "triton": tri.at[idx, c],
                    "raw": raw.at[idx, c],
                })

    float_mismatch_counts = {}
    for c in common_float_cols:
        a = pd.to_numeric(tri[c].iloc[:n], errors="coerce")
        b = pd.to_numeric(raw[c].iloc[:n], errors="coerce")
        diff = (a - b).abs()
        mask = diff > 1e-4
        count = int(mask.sum())
        float_mismatch_counts[c] = count

        if count:
            idxs = list(mask[mask].index[:50])
            for idx in idxs:
                mismatches.append({
                    "row": int(idx),
                    "column": c,
                    "triton": tri.at[idx, c],
                    "raw": raw.at[idx, c],
                    "abs_diff": float(diff.iloc[idx]),
                })

    mismatch_df = pd.DataFrame(mismatches[:300])
    mismatch_df.to_csv(MISMATCH_CSV, index=False)

    report["compared_exact_columns"] = common_key_cols
    report["compared_float_columns"] = common_float_cols
    report["exact_mismatch_counts"] = exact_mismatch_counts
    report["float_mismatch_counts_tol_1e-4"] = float_mismatch_counts
    report["total_mismatch_cells"] = int(
        sum(exact_mismatch_counts.values()) + sum(float_mismatch_counts.values())
    )
    report["mismatch_sample_csv"] = str(MISMATCH_CSV)

    selected_rows = sorted(set([0, 1, 2, 10, 100, 500, 1000, 2000, 4500, 8999]))
    selected_rows = [r for r in selected_rows if r < n]

    selected = []
    for r in selected_rows:
        item = {"row": int(r)}
        for c in common_key_cols + common_float_cols:
            item[f"triton_{c}"] = tri.at[r, c]
            item[f"raw_{c}"] = raw.at[r, c]
        selected.append(item)

    report["selected_row_spot_check"] = selected

    return report


def compare_jsons(tri_json: Path, raw_json: Path):
    tri = json.loads(tri_json.read_text())
    raw = json.loads(raw_json.read_text())

    metric_keys = [
        "frames_processed",
        "yolopx_forward_ms_per_frame",
        "depth_forward_ms_per_frame",
        "traffic_forward_ms_per_frame",
        "combined_forward_ms_per_frame",
        "yolopx_forward_fps",
        "depth_forward_fps",
        "traffic_forward_fps",
        "combined_forward_fps",
        "end_to_end_ms_per_frame",
        "end_to_end_fps",
        "read_ms_per_frame",
        "preprocess_ms_per_frame",
        "yolopx_nms_ms_per_frame",
        "lane_post_and_decision_ms_per_frame",
        "traffic_total_call_ms_per_frame",
        "traffic_decision_only_ms_per_frame",
        "draw_ms_per_frame",
        "write_ms_per_frame",
    ]

    metrics = {}
    for k in metric_keys:
        tv = tri.get(k)
        rv = raw.get(k)
        tf = safe_float(tv)
        rf = safe_float(rv)
        diff = None
        ratio = None
        if tf is not None and rf is not None:
            diff = rf - tf
            ratio = rf / tf if tf != 0 else None
        metrics[k] = {
            "triton": tv,
            "raw": rv,
            "raw_minus_triton": diff,
            "raw_over_triton": ratio,
        }

    core_keys = [
        "source_resolution",
        "source_fps_metadata",
        "total_frames_in_video",
        "start_frame",
        "end_frame_exclusive",
        "frames_processed",
        "lane_strategy",
    ]

    core = {}
    for k in core_keys:
        core[k] = {
            "triton": tri.get(k),
            "raw": raw.get(k),
            "same": tri.get(k) == raw.get(k),
        }

    return {
        "core": core,
        "fps_and_timing_metrics": metrics,
        "triton_summary": str(tri_json),
        "raw_summary": str(raw_json),
    }



def json_safe(obj):
    """Convert pandas/numpy scalar values into normal Python JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if pd.isna(obj) if not isinstance(obj, (dict, list, tuple)) else False:
        return None
    return obj

def write_report(outputs, csv_report, json_report):
    final = {
        "files": {
            "triton_csv": str(outputs["tri_csv"]),
            "raw_csv": str(outputs["raw_csv"]),
            "triton_json": str(outputs["tri_json"]),
            "raw_json": str(outputs["raw_json"]),
            "triton_mp4": str(outputs["tri_mp4"]),
            "raw_mp4": str(outputs["raw_mp4"]),
            "triton_mp4_size_mb": round(outputs["tri_mp4"].stat().st_size / 1024 / 1024, 2),
            "raw_mp4_size_mb": round(outputs["raw_mp4"].stat().st_size / 1024 / 1024, 2),
            "triton_mp4_sha256": sha256(outputs["tri_mp4"]),
            "raw_mp4_sha256": sha256(outputs["raw_mp4"]),
            "raw_pipeline": str(RAW_PIPELINE),
            "raw_pipeline_sha256": sha256(RAW_PIPELINE),
            "triton_pipeline": str(TRITON_PIPELINE),
            "triton_pipeline_sha256": sha256(TRITON_PIPELINE),
        },
        "csv_comparison": csv_report,
        "json_comparison": json_report,
    }

    REPORT_JSON.write_text(json.dumps(json_safe(final), indent=2))

    tri_fps = json_report["fps_and_timing_metrics"]["end_to_end_fps"]["triton"]
    raw_fps = json_report["fps_and_timing_metrics"]["end_to_end_fps"]["raw"]
    tri_e2e_ms = json_report["fps_and_timing_metrics"]["end_to_end_ms_per_frame"]["triton"]
    raw_e2e_ms = json_report["fps_and_timing_metrics"]["end_to_end_ms_per_frame"]["raw"]

    md = []
    md.append("# Triton vs Raw TensorRT 5-Minute Comparison")
    md.append("")
    md.append("## Main result")
    md.append("")
    md.append(f"- Triton end-to-end FPS: `{tri_fps}`")
    md.append(f"- Raw local TensorRT end-to-end FPS: `{raw_fps}`")
    md.append(f"- Triton end-to-end ms/frame: `{tri_e2e_ms}`")
    md.append(f"- Raw local TensorRT end-to-end ms/frame: `{raw_e2e_ms}`")
    md.append("")
    md.append("## CSV result equality")
    md.append("")
    md.append(f"- Triton rows: `{csv_report['triton_rows']}`")
    md.append(f"- Raw rows: `{csv_report['raw_rows']}`")
    md.append(f"- Total mismatch cells: `{csv_report['total_mismatch_cells']}`")
    md.append(f"- Mismatch sample CSV: `{csv_report['mismatch_sample_csv']}`")
    md.append("")
    md.append("## Exact mismatch counts")
    md.append("")
    for k, v in csv_report["exact_mismatch_counts"].items():
        md.append(f"- `{k}`: `{v}`")
    md.append("")
    md.append("## Float mismatch counts, tolerance 1e-4")
    md.append("")
    for k, v in csv_report["float_mismatch_counts_tol_1e-4"].items():
        md.append(f"- `{k}`: `{v}`")
    md.append("")
    md.append("## Output files")
    md.append("")
    for k, v in final["files"].items():
        md.append(f"- `{k}`: `{v}`")
    md.append("")

    REPORT_MD.write_text("\n".join(md))

    print("\n" + "=" * 90)
    print("FINAL COMPARISON REPORT")
    print("=" * 90)
    print(REPORT_MD.read_text())
    print("JSON report:", REPORT_JSON)
    print("Markdown report:", REPORT_MD)


def main():
    print("=== PRECHECK ===")
    for label, path in [
        ("video", VIDEO),
        ("YOLOPX engine", YOLOPX_ENGINE),
        ("Depth engine", DEPTH_ENGINE),
        ("Traffic engine", TRAFFIC_ENGINE),
        ("Triton pipeline", TRITON_PIPELINE),
    ]:
        assert_exists(path, label)
        print(label, "OK:", path)

    print("\nEngine hashes:")
    print("YOLOPX :", sha256(YOLOPX_ENGINE))
    print("Depth  :", sha256(DEPTH_ENGINE))
    print("Traffic:", sha256(TRAFFIC_ENGINE))
    print("Video  :", sha256(VIDEO))

    # FINAL_LOCK:
    # final_rawtrt_preserved_pipeline.py is now the locked apple-to-apple RawTRT file.
    # Do not regenerate it from final_triton_preserved_pipeline.py unless it is missing.
    if RAW_PIPELINE.exists():
        print("Using existing locked raw pipeline:", RAW_PIPELINE)
        run([sys.executable, "-m", "py_compile", str(RAW_PIPELINE)], check=True)
    else:
        print("Raw pipeline missing; generating fallback raw pipeline.")
        create_raw_pipeline()

    run_pipeline_pair()

    outputs = load_outputs()

    print("\n=== OUTPUTS FOUND ===")
    for k, v in outputs.items():
        print(k, "=>", v, "size_MB=", round(v.stat().st_size / 1024 / 1024, 2))

    csv_report = compare_csvs(outputs["tri_csv"], outputs["raw_csv"])
    json_report = compare_jsons(outputs["tri_json"], outputs["raw_json"])
    write_report(outputs, csv_report, json_report)


if __name__ == "__main__":
    main()
