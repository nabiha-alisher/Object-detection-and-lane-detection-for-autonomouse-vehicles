import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/fyp")
OUT = ROOT / "output"
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

# Original baseline (unpatched lane code) - kept for apples-to-apples vs your existing matrix
# Lane-fixed variants (_v2 suffix) - vectorized _find_peaks_strict, verified bit-identical decisions
# rawtrt_seq - no-Triton raw TensorRT path
VARIANTS = [
    ("http_seq",          ROOT / "fps_http_seq.py",          "fps_http_seq"),
    ("grpc_seq",          ROOT / "fps_grpc_seq.py",          "fps_grpc_seq"),
    ("http_parallel",     ROOT / "fps_http_parallel.py",     "fps_http_parallel"),
    ("grpc_parallel",     ROOT / "fps_grpc_parallel.py",     "fps_grpc_parallel"),
    ("rawtrt_seq",        ROOT / "fps_rawtrt_seq.py",        "fps_rawtrt_seq"),

    ("http_seq_v2",       ROOT / "fps_http_seq_v2.py",       "fps_http_seq_v2"),
    ("grpc_seq_v2",       ROOT / "fps_grpc_seq_v2.py",       "fps_grpc_seq_v2"),
    ("http_parallel_v2",  ROOT / "fps_http_parallel_v2.py",  "fps_http_parallel_v2"),
    ("grpc_parallel_v2",  ROOT / "fps_grpc_parallel_v2.py",  "fps_grpc_parallel_v2"),
    ("rawtrt_seq_v2",     ROOT / "fps_rawtrt_seq_v2.py",     "fps_rawtrt_seq_v2"),
]

def run_one(name, script, suffix, minutes):
    if not script.exists():
        raise FileNotFoundError(script)

    env = os.environ.copy()
    env["FPS_TEST_MINUTES"] = str(minutes)
    env["FPS_NAME_SUFFIX"] = suffix
    env["FPS_LANE_STRATEGY"] = "corridor_memory_two_hits"
    env["TRITON_HTTP_URL"] = env.get("TRITON_HTTP_URL", "localhost:8000")
    env["TRITON_GRPC_URL"] = env.get("TRITON_GRPC_URL", "localhost:8001")
    env["PYTHONUNBUFFERED"] = "1"

    log_path = LOGS / f"fps_{name}_{minutes}min.log"

    print("\n" + "=" * 80)
    print(f"RUNNING {name}")
    print(f"script : {script}")
    print(f"minutes: {minutes}")
    print(f"log    : {log_path}")
    print("=" * 80)

    t0 = time.time()

    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.Popen(
            [sys.executable, "-u", str(script)],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            log.write(line)

        rc = p.wait()

    elapsed = time.time() - t0

    if rc != 0:
        raise RuntimeError(f"{name} failed with exit code {rc}. See {log_path}")

    candidates = sorted(
        OUT.rglob(f"*_{suffix}_summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(f"No summary JSON found for suffix {suffix}")

    summary_path = candidates[0]
    data = json.loads(summary_path.read_text())

    row = {
        "variant": name,
        "minutes": minutes,
        "summary": str(summary_path),
        "elapsed_wall_sec": round(elapsed, 3),
        "frames": data.get("frames_processed"),
        "e2e_fps": data.get("end_to_end_fps"),
        "e2e_ms": data.get("end_to_end_ms_per_frame"),
        "combined_forward_ms": data.get("combined_forward_ms_per_frame"),
        "yolopx_ms": data.get("yolopx_forward_ms_per_frame"),
        "depth_ms": data.get("depth_forward_ms_per_frame"),
        "traffic_ms": data.get("traffic_forward_ms_per_frame"),
        "traffic_total_call_ms": data.get("traffic_total_call_ms_per_frame"),
        "lane_ms": data.get("lane_post_and_decision_ms_per_frame"),
        "draw_ms": data.get("draw_ms_per_frame"),
        "write_ms": data.get("write_ms_per_frame"),
        "csv": str(summary_path).replace("_summary.json", "_frames.csv"),
    }

    print("\nRESULT:", json.dumps(row, indent=2))
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=0.10)
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=[v[0] for v in VARIANTS],
        help="Optional subset, e.g.: --only rawtrt_seq  OR  --only http_seq_v2 grpc_seq_v2 rawtrt_seq_v2",
    )
    args = ap.parse_args()

    selected = VARIANTS
    if args.only:
        selected = [v for v in VARIANTS if v[0] in args.only]

    results = []

    for name, script, suffix in selected:
        results.append(run_one(name, script, suffix, args.minutes))

    tag = "_".join(r["variant"] for r in results) if args.only else "all"
    report_path = OUT / f"fps_matrix_v2_{tag}_{args.minutes}min_report.json"
    report_path.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 80)
    print("FINAL FPS MATRIX")
    print("=" * 80)
    print(f"{'variant':20s} {'FPS':>10s} {'E2E ms':>10s} {'model ms':>10s} {'lane ms':>10s} {'write ms':>10s}")
    for r in results:
        print(
            f"{r['variant']:20s} "
            f"{float(r['e2e_fps']):10.3f} "
            f"{float(r['e2e_ms']):10.3f} "
            f"{float(r['combined_forward_ms']):10.3f} "
            f"{float(r['lane_ms']):10.3f} "
            f"{float(r['write_ms']):10.3f}"
        )

    print("\nSaved report:", report_path)

if __name__ == "__main__":
    main()
