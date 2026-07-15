# RunPod Setup Guide

This document explains how to restore and run the final FYP AI traffic assistance pipeline on RunPod.

## 1. Recommended RunPod Environment

Use an NVIDIA GPU pod with an RTX 4090 or equivalent GPU. The tested setup used an NVIDIA Triton container:

```bash
nvcr.io/nvidia/tritonserver:24.05-py3
```

Recommended pod settings:

```text
GPU: RTX 4090
Workspace: /workspace
Start command: bash -c "sleep infinity"
```

Required exposed ports:

```text
9000  FastAPI dashboard
8000  Triton HTTP endpoint
8001  Triton gRPC endpoint
8002  Triton metrics endpoint
22    SSH/SCP access
```

## 2. Final Important Files

The final locked project files are:

```text
final_triton_preserved_pipeline.py
final_rawtrt_preserved_pipeline.py
app/main.py
```

The optimized profiler/test files are:

```text
fps_http_seq.py
fps_grpc_seq.py
fps_http_parallel.py
fps_grpc_parallel.py
fps_grpc_parallel_v2.py
fps_rawtrt_seq_v2.py
```

Meaning of the main optimized files:

```text
fps_grpc_parallel_v2.py = Triton gRPC + parallel inference + vectorized lane logic
fps_rawtrt_seq_v2.py   = Raw TensorRT + vectorized lane logic
```

## 3. Large Files Not Stored in GitHub

Do not push the following files to GitHub:

```text
*.engine
*.onnx
*.pth
*.pt
*.mp4
large CSV outputs
logs/
output/
datasets/
checkpoints/
large zip evidence packages
```

These should be stored in Google Drive or another external storage location.

Expected local folders after downloading assets:

```text
engines/
onnx/
weights/
assets/
triton_model_repo/
```

## 4. Install Python Dependencies

From the repository root:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install --force-reinstall --no-deps numpy==1.26.4
```

The final NumPy reinstall is important because the tested TensorRT/PyCUDA setup required the NumPy 1.x ABI.

## 5. Start Triton Server

Start Triton using the model repository:

```bash
cd /workspace/fyp
mkdir -p logs
nohup tritonserver \
  --model-repository=/workspace/fyp/triton_model_repo \
  --http-port=8000 \
  --grpc-port=8001 \
  --metrics-port=8002 \
  --log-verbose=1 \
  > logs/triton_server.log 2>&1 &
```

Check readiness:

```bash
curl -s -o /dev/null -w "Triton ready: %{http_code}\n" localhost:8000/v2/health/ready
curl -s -o /dev/null -w "YOLOPX ready: %{http_code}\n" localhost:8000/v2/models/yolopx/ready
curl -s -o /dev/null -w "Depth ready: %{http_code}\n" localhost:8000/v2/models/depth/ready
curl -s -o /dev/null -w "Traffic ready: %{http_code}\n" localhost:8000/v2/models/traffic/ready
```

Expected result:

```text
Triton ready: 200
YOLOPX ready: 200
Depth ready: 200
Traffic ready: 200
```

## 6. Run the Dashboard

The dashboard is launched from:

```text
app/main.py
```

Run:

```bash
cd /workspace/fyp
python3 app/main.py
```

Open the dashboard using the RunPod public URL mapped to port 9000.

Important note: the locked dashboard/live preview is HTTP-based. The gRPC and vectorized-lane tests were profiler/optimization variants unless intentionally ported into the dashboard.

## 7. Run FPS Profilers

Original four profiler comparison:

```bash
cd /workspace/fyp
python3 scripts/run_fps_variants_matrix.py --minutes 5.0
```

Latest gRPC parallel + vectorized lane profiler:

```bash
cd /workspace/fyp
mkdir -p logs
env FPS_TEST_MINUTES=5.0 FPS_NAME_SUFFIX=fps_grpc_parallel_v2 FPS_LANE_STRATEGY=corridor_memory_two_hits PYTHONUNBUFFERED=1 python3 -u fps_grpc_parallel_v2.py 2>&1 | tee logs/round2_grpc_parallel_v2_vectorized_lane_5min.log
```

Raw TensorRT + vectorized lane profiler:

```bash
cd /workspace/fyp
mkdir -p logs
env FPS_TEST_MINUTES=5.0 FPS_NAME_SUFFIX=fps_rawtrt_seq_v2 FPS_LANE_STRATEGY=corridor_memory_two_hits PYTHONUNBUFFERED=1 python3 -u fps_rawtrt_seq_v2.py 2>&1 | tee logs/round2_rawtrt_seq_v2_vectorized_lane_5min.log
```

## 8. Verification Commands

Verify vectorized lane peak logic:

```bash
cd /workspace/fyp
python3 verify_find_peaks.py
```

Compare CSV decision outputs:

```bash
cd /workspace/fyp
python3 compare_variants.py --output-dir output/FINAL_3MODEL_UNIFIED --baseline fps_http_seq
```

For the specific old gRPC parallel vs vectorized gRPC parallel audit, compare:

```text
vid10min_3model_full_fps_grpc_parallel_frames.csv
vid10min_3model_full_fps_grpc_parallel_v2_frames.csv
```

The verified decision-level comparison over selected columns showed 0 mismatched decision rows over 9000 frames.
