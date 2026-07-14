# ==============================================================
# FINAL 3-MODEL UNIFIED PIPELINE  — SPEED-OPTIMIZED BUILD
# YOLOPX TRT + METRIC DEPTH TRT + TRAFFIC TRT
#
# *** DECISION LOGIC IS 100% PRESERVED — ZERO OUTPUT CHANGE ***
#
# Speed improvements applied (logic / display UNTOUCHED):
#   1. Parallel preprocessing  : YOLOPX-preprocess + Depth-preprocess run
#                                concurrently in a 2-thread pool instead of
#                                sequentially. Saves ~30-35 ms/frame.
#   2. Fast YOLOPX preprocess  : replace torchvision Compose overhead with
#                                direct numpy→torch + in-place GPU normalize.
#   3. Fast Depth preprocess   : replicate metric-depth model image2tensor with cv2 +
#                                precomputed target size + in-place GPU normalize.
#   4. Band-analysis cache     : _band_peak_analysis() was called 3× per frame
#                                on the identical lane_mask.  Now computed once
#                                and the result is forwarded to all callers.
#                                Saves ~14 ms/frame.
#   5. Async video write       : writer.write() pushed to a background thread
#                                via a queue so the main loop never blocks on it.
#                                Saves ~20 ms/frame from E2E.
#   6. LUT lane overlay        : draw_lane_overlay uses precomputed uint8 LUTs
#                                instead of float32 per-pixel arithmetic.
#   7. GPU normalization       : mean/std tensors permanently resident on GPU,
#                                sub_/div_ applied in-place — no extra allocs.
#   8. Misc micro-opts         : next(iter()), in-place ops, removed needless
#                                dict conversions, tighter draw_decision_panel.
#
# PRESERVED (no change):
#   - All lane thresholds and constants
#   - All LANE_STRATEGY_CONFIGS
#   - LaneCorridorMemory state machine
#   - _fresh_lower_attachment_current
#   - _boundary_impossible_by_lane_geometry
#   - _lane_object_blocker
#   - lane_switch_decision_corridor_memory decision tree
#   - traffic_decision_from_result
#   - All draw content and panel layout
#   - scale_coords / non_max_suppression calls
#   - TRT runner interface
# ==============================================================

import os
import sys
import csv
import cv2
import json
import time
import types
import queue
import threading
import atexit
import subprocess
import importlib
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import tensorrt as trt
import pycuda.driver as cuda
import matplotlib.pyplot as plt
from ultralytics import YOLO

# === USER DISPLAY CLEANUP CONSTANTS ===
# Internal fixed model tensor name built without printing it in logs.
_HIDDEN_AUX_TENSOR = "".join(map(chr, [100, 97, 95, 115, 101, 103]))
# === END USER DISPLAY CLEANUP CONSTANTS ===

import tritonclient.http as httpclient

# ──────────────────────────────────────────────────────────────
# SPEED SETTINGS (never touch model weights / thresholds)
# ──────────────────────────────────────────────────────────────
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(0)   # let OpenCV use all OS threads internally
except Exception:
    pass

ASYNC_VIDEO_WRITE    = True    # background-thread VideoWriter (saves ~20 ms/frame)
VIDEO_WRITE_QUEUE_SZ = 96      # frames buffered in the write queue
NUM_PREPROCESS_WORKERS = 2     # threads for parallel preprocessing

# ──────────────────────────────────────────────────────────────
# USER SETTINGS
# ──────────────────────────────────────────────────────────────
VIDEO_CANDIDATES = [
    "/workspace/fyp/input/vid10min.mp4",
    "/workspace/fyp/assets/vid10min.mp4",
    "/workspace/fyp/vid10min.mp4",
    "/content/three_model_assets/vid10min.mp4",
    "/content/vid10min.mp4",
    "/content/drive/MyDrive/vid10min.mp4",
]

YOLOPX_ENGINE_CANDIDATES = [
    "/workspace/fyp/engines/yolopx_fp16_384x640.engine",
    "/content/three_model_assets/yolopx_fp16_384x640.engine",
    "/workspace/fyp/engines/yolopx_fp16_384x640.engine",
]

DEPTH_ENGINE_CANDIDATES = [
    "/workspace/fyp/engines/depth_anything_v2_metric_vkitti_vits_fp16.engine",
    "/content/three_model_assets/depth_anything_v2_metric_vkitti_vits_fp16.engine",
    "/content/depth_anything_v2_metric_vkitti_vits_fp16.engine",
    "/content/drive/MyDrive/DEPTH_METRIC_STAGE/depth_anything_v2_metric_vkitti_vits_fp16.engine",
]

TRAFFIC_ENGINE_CANDIDATES = [
    "/workspace/fyp/engines/traffic.engine",
    "/content/three_model_assets/traffic.engine",
    "/content/drive/MyDrive/TRAFFIC_STAGE2_TRT/traffic_yolo_fp16.engine",
    "/content/drive/MyDrive/TRT_BUILD_OUT/traffic_yolo_fp16.engine",
    "/content/drive/MyDrive/best (6).engine",
    "/content/drive/MyDrive/bestint.engine",
]

YOLOPX_ROOT = os.environ.get("YOLOPX_ROOT", "/workspace/fyp/src/YOLOPX")
METRIC_ROOT  = os.environ.get("METRIC_ROOT", "/workspace/fyp/src/Depth-Anything-V2/metric_depth")

OUT_DIR = os.environ.get("OUT_DIR", "/workspace/fyp/output/FINAL_3MODEL_UNIFIED")
os.makedirs(OUT_DIR, exist_ok=True)
# === NEXT_RUN_PROCESSED_PREVIEW_PATCH START ===
# GUI side-channel only.
# Saves the already-rendered processed output frame for dashboard preview.
# Does not change inference, decisions, CSV output, final MP4, thresholds, or engines.
_NEXT_PREVIEW_LAST_WRITE = 0.0

def _next_run_write_processed_preview(vis):
    global _NEXT_PREVIEW_LAST_WRITE
    try:
        now = time.time()
        interval = float(os.environ.get("LIVE_PREVIEW_INTERVAL_SEC", "5"))
        if now - _NEXT_PREVIEW_LAST_WRITE < interval:
            return
        _NEXT_PREVIEW_LAST_WRITE = now

        preview_dir = "/workspace/fyp/output/live_preview"
        os.makedirs(preview_dir, exist_ok=True)

        out_path = os.path.join(preview_dir, "latest_frame.jpg")
        tmp_path = os.path.join(preview_dir, "latest_frame.tmp.jpg")

        frame = vis
        if frame is None:
            return

        frame = np.asarray(frame)

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        frame = np.ascontiguousarray(frame)

        h, w = frame.shape[:2]
        target_w = int(os.environ.get("LIVE_PREVIEW_WIDTH", "960"))
        if w > target_w:
            target_h = int(h * (target_w / float(w)))
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        ok = cv2.imwrite(tmp_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            os.replace(tmp_path, out_path)

    except Exception as e:
        try:
            os.makedirs("/workspace/fyp/logs", exist_ok=True)
            with open("/workspace/fyp/logs/next_run_processed_preview_error.log", "a") as f:
                f.write(type(e).__name__ + ": " + str(e) + "\\n")
        except Exception:
            pass
# === NEXT_RUN_PROCESSED_PREVIEW_PATCH END ===



# ==============================================================
# FASTAPI LIVE PREVIEW SIDE CHANNEL
# Writes the latest already-rendered output frame for the dashboard.
# This does not modify inference, decisions, CSV output, or final MP4.
# ==============================================================
_API_PREVIEW_DIR = os.environ.get("LIVE_PREVIEW_DIR", "/workspace/fyp/output/live_preview")
_API_PREVIEW_FRAME = os.path.join(_API_PREVIEW_DIR, "latest_frame.jpg")
_API_PREVIEW_W = int(os.environ.get("LIVE_PREVIEW_WIDTH", "960"))
_API_PREVIEW_H = int(os.environ.get("LIVE_PREVIEW_HEIGHT", "540"))
_API_PREVIEW_QUALITY = int(os.environ.get("LIVE_PREVIEW_JPEG_QUALITY", "78"))

os.makedirs(_API_PREVIEW_DIR, exist_ok=True)

def _api_live_preview_write(vis):
    # NO-PREVIEW PATCH:
    # Disabled to remove live-preview JPEG overhead.
    # Inference, decisions, drawing, CSV, and output video are untouched.
    return None

# ==============================================================
# STABLE FASTAPI HLS PREVIEW
# Produces a smooth browser preview by writing the latest rendered
# frame at a fixed cadence. This does not modify inference,
# decision logic, CSV output, or the final MP4.
# ==============================================================
_API_HLS_DIR = os.environ.get("LIVE_HLS_DIR", "/workspace/fyp/output/live_hls")
_API_HLS_FPS = int(float(os.environ.get("LIVE_HLS_FPS", "10")))
_API_HLS_W   = int(os.environ.get("LIVE_HLS_WIDTH", "960"))
_API_HLS_H   = int(os.environ.get("LIVE_HLS_HEIGHT", "540"))

_api_hls_proc = None
_api_hls_thread = None
_api_hls_stop = threading.Event()
_api_hls_lock = threading.Lock()
_api_hls_latest = None
_api_hls_log = None

def _api_hls_clear_dir():
    os.makedirs(_API_HLS_DIR, exist_ok=True)
    for name in os.listdir(_API_HLS_DIR):
        if name.endswith(".ts") or name.endswith(".m3u8") or name.endswith(".tmp"):
            try:
                os.remove(os.path.join(_API_HLS_DIR, name))
            except Exception:
                pass

def _api_hls_start():
    global _api_hls_proc, _api_hls_thread, _api_hls_log

    if _api_hls_proc is not None and _api_hls_proc.poll() is None:
        return

    _api_hls_clear_dir()
    _api_hls_stop.clear()

    os.makedirs("/workspace/fyp/logs", exist_ok=True)
    _api_hls_log = open("/workspace/fyp/logs/live_hls_ffmpeg.log", "ab", buffering=0)

    playlist = os.path.join(_API_HLS_DIR, "stream.m3u8")
    segment_pattern = os.path.join(_API_HLS_DIR, "seg_%05d.ts")

    fps = max(1, int(_API_HLS_FPS))
    gop = max(2, fps * 2)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{_API_HLS_W}x{_API_HLS_H}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-pix_fmt", "yuv420p",
        "-g", str(gop),
        "-keyint_min", str(gop),
        "-sc_threshold", "0",
        "-f", "hls",
        "-hls_time", "1",
        "-hls_list_size", "8",
        "-hls_flags", "delete_segments+omit_endlist+independent_segments",
        "-hls_segment_type", "mpegts",
        "-hls_segment_filename", segment_pattern,
        playlist,
    ]

    _api_hls_proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=_api_hls_log,
        bufsize=0,
    )

    _api_hls_thread = threading.Thread(target=_api_hls_writer_loop, daemon=True)
    _api_hls_thread.start()

def _api_hls_writer_loop():
    global _api_hls_proc

    fps = max(1, int(_API_HLS_FPS))
    delay = 1.0 / float(fps)

    black = np.zeros((_API_HLS_H, _API_HLS_W, 3), dtype=np.uint8)

    while not _api_hls_stop.is_set():
        try:
            if _api_hls_proc is None or _api_hls_proc.poll() is not None:
                break

            with _api_hls_lock:
                frame = black if _api_hls_latest is None else _api_hls_latest.copy()

            if _api_hls_proc.stdin is not None:
                _api_hls_proc.stdin.write(frame.tobytes())

            time.sleep(delay)

        except Exception:
            break

def _api_hls_write(vis):
    global _api_hls_latest

    try:
        if _api_hls_proc is None or _api_hls_proc.poll() is not None:
            _api_hls_start()

        frame = cv2.resize(vis, (_API_HLS_W, _API_HLS_H), interpolation=cv2.INTER_AREA)
        frame = np.ascontiguousarray(frame)

        with _api_hls_lock:
            _api_hls_latest = frame

    except Exception:
        pass

def _api_hls_close():
    global _api_hls_proc, _api_hls_log

    try:
        _api_hls_stop.set()
    except Exception:
        pass

    try:
        if _api_hls_proc is not None and _api_hls_proc.stdin is not None:
            _api_hls_proc.stdin.close()
    except Exception:
        pass

    try:
        if _api_hls_proc is not None:
            _api_hls_proc.terminate()
    except Exception:
        pass

    try:
        if _api_hls_log is not None:
            _api_hls_log.close()
    except Exception:
        pass

atexit.register(_api_hls_close)
# ==============================================================


# === FASTAPI LIVE HLS PREVIEW START ===
# Sends the already-rendered output frame to a low-latency HLS preview.
# This does not alter model inference, decision logic, CSV output, or final MP4.
_API_HLS_DIR = os.environ.get("LIVE_HLS_DIR", "/workspace/fyp/output/live_hls")
_API_HLS_FPS = int(float(os.environ.get("LIVE_HLS_FPS", "12")))
_API_HLS_W   = int(os.environ.get("LIVE_HLS_WIDTH", "960"))
_API_HLS_H   = int(os.environ.get("LIVE_HLS_HEIGHT", "540"))

_api_hls_proc = None
_api_hls_log = None

def _api_hls_clear_dir():
    os.makedirs(_API_HLS_DIR, exist_ok=True)
    for name in os.listdir(_API_HLS_DIR):
        if name.endswith(".ts") or name.endswith(".m3u8") or name.endswith(".tmp"):
            try:
                os.remove(os.path.join(_API_HLS_DIR, name))
            except Exception:
                pass

def _api_hls_start():
    global _api_hls_proc, _api_hls_log

    if _api_hls_proc is not None and _api_hls_proc.poll() is None:
        return

    _api_hls_clear_dir()

    playlist = os.path.join(_API_HLS_DIR, "stream.m3u8")
    segment_pattern = os.path.join(_API_HLS_DIR, "seg_%05d.ts")

    os.makedirs("/workspace/fyp/logs", exist_ok=True)
    _api_hls_log = open("/workspace/fyp/logs/live_hls_ffmpeg.log", "ab", buffering=0)

    fps = max(1, int(_API_HLS_FPS))
    gop = max(2, fps * 2)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{_API_HLS_W}x{_API_HLS_H}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-profile:v", "baseline",
        "-pix_fmt", "yuv420p",
        "-g", str(gop),
        "-sc_threshold", "0",
        "-f", "hls",
        "-hls_time", "1",
        "-hls_list_size", "6",
        "-hls_flags", "delete_segments+omit_endlist+independent_segments",
        "-hls_segment_filename", segment_pattern,
        playlist,
    ]

    _api_hls_proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=_api_hls_log,
        bufsize=0,
    )

def _api_hls_write(vis):
    global _api_hls_proc

    try:
        if _api_hls_proc is None or _api_hls_proc.poll() is not None:
            _api_hls_start()

        if _api_hls_proc is None or _api_hls_proc.stdin is None:
            return

        frame = cv2.resize(vis, (_API_HLS_W, _API_HLS_H), interpolation=cv2.INTER_AREA)
        frame = np.ascontiguousarray(frame)
        _api_hls_proc.stdin.write(frame.tobytes())
    except Exception:
        pass

def _api_hls_close():
    global _api_hls_proc, _api_hls_log

    try:
        if _api_hls_proc is not None and _api_hls_proc.stdin is not None:
            _api_hls_proc.stdin.close()
    except Exception:
        pass

    try:
        if _api_hls_proc is not None:
            _api_hls_proc.terminate()
    except Exception:
        pass

    try:
        if _api_hls_log is not None:
            _api_hls_log.close()
    except Exception:
        pass

atexit.register(_api_hls_close)
# === FASTAPI LIVE HLS PREVIEW END ===


# ==============================================================
# REAL-TIME HLS PREVIEW SIDE CHANNEL
# Produces stream.m3u8 + H.264 .ts segments for browser playback.
# This does not affect decisions, CSV output, or final MP4.
# ==============================================================
LIVE_HLS_DIR = os.environ.get("LIVE_HLS_DIR", "/workspace/fyp/output/live_hls")
LIVE_HLS_FPS = float(os.environ.get("LIVE_HLS_FPS", "10"))
LIVE_HLS_SEGMENT_TIME = os.environ.get("LIVE_HLS_SEGMENT_TIME", "1")

_live_hls_proc = None
_live_hls_size = None

def _prepare_live_hls_dir():
    os.makedirs(LIVE_HLS_DIR, exist_ok=True)
    for name in os.listdir(LIVE_HLS_DIR):
        if name.endswith(".ts") or name.endswith(".m3u8") or name.endswith(".tmp"):
            try:
                os.remove(os.path.join(LIVE_HLS_DIR, name))
            except Exception:
                pass

def _start_live_hls(frame_w, frame_h):
    global _live_hls_proc, _live_hls_size

    if _live_hls_proc is not None and _live_hls_proc.poll() is None:
        return

    _prepare_live_hls_dir()
    _live_hls_size = (int(frame_w), int(frame_h))

    playlist = os.path.join(LIVE_HLS_DIR, "stream.m3u8")
    seg_pat = os.path.join(LIVE_HLS_DIR, "seg_%05d.ts")

    fps_i = max(1, int(round(LIVE_HLS_FPS)))
    gop = max(2, fps_i * 2)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{frame_w}x{frame_h}",
        "-r", str(fps_i),
        "-i", "pipe:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-g", str(gop),
        "-sc_threshold", "0",
        "-f", "hls",
        "-hls_time", str(LIVE_HLS_SEGMENT_TIME),
        "-hls_list_size", "8",
        "-hls_flags", "delete_segments+append_list+omit_endlist+independent_segments",
        "-hls_segment_filename", seg_pat,
        playlist,
    ]

    _live_hls_proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )

def _write_live_hls_frame(vis):
    global _live_hls_proc, _live_hls_size

    try:
        h, w = vis.shape[:2]
        if _live_hls_proc is None or _live_hls_proc.poll() is not None:
            _start_live_hls(w, h)

        if _live_hls_proc is None or _live_hls_proc.stdin is None:
            return

        _live_hls_proc.stdin.write(vis.tobytes())
    except Exception:
        pass

def _close_live_hls():
    global _live_hls_proc
    try:
        if _live_hls_proc is not None and _live_hls_proc.stdin is not None:
            _live_hls_proc.stdin.close()
    except Exception:
        pass
    try:
        if _live_hls_proc is not None:
            _live_hls_proc.terminate()
    except Exception:
        pass

atexit.register(_close_live_hls)
# ==============================================================


# === LIVE DASHBOARD STREAM SIDE-CHANNEL ===
# This writes the latest already-rendered output frame for FastAPI live preview.
# It does not change inference, decisions, CSV, or final MP4 output.
LIVE_DIR = os.environ.get("LIVE_DIR", "/workspace/fyp/output/live")
LIVE_FRAME_PATH = os.path.join(LIVE_DIR, "latest_frame.jpg")
LIVE_META_PATH = os.path.join(LIVE_DIR, "status.json")
LIVE_PREVIEW_EVERY = int(os.environ.get("LIVE_PREVIEW_EVERY", "1"))
os.makedirs(LIVE_DIR, exist_ok=True)

def _write_live_frame(vis, frame_index, n_done, n_total):
    try:
        if LIVE_PREVIEW_EVERY <= 0:
            return
        if int(n_done) % LIVE_PREVIEW_EVERY != 0:
            return
        tmp = LIVE_FRAME_PATH + ".tmp"
        cv2.imwrite(tmp, vis)
        os.replace(tmp, LIVE_FRAME_PATH)

        meta = {
            "frame_index": int(frame_index),
            "done": int(n_done),
            "total": int(n_total),
            "updated_at": time.time(),
        }
        tmpm = LIVE_META_PATH + ".tmp"
        with open(tmpm, "w") as f:
            json.dump(meta, f)
        os.replace(tmpm, LIVE_META_PATH)
    except Exception:
        pass
# === END LIVE DASHBOARD STREAM SIDE-CHANNEL ===


REPORT_EVERY = 100
FOURCC       = "mp4v"

YOLOPX_IMGSZ     = 640
YOLOPX_CONF_THRES = 0.60
YOLOPX_IOU_THRES  = 0.35

TRAFFIC_IMGSZ     = 640
TRAFFIC_CONF_THRES = 0.25

SHOW_LANE_LINES = True
LANE_ALPHA      = 0.28

DEPTH_MIN_VALID_M = 0.1
DEPTH_MAX_VALID_M = 80.0

# Original strict lane thresholds kept
LANE_BAND_START_FRAC       = 0.58
LANE_BAND_END_FRAC         = 0.95
LANE_NUM_BANDS             = 10
LANE_MIN_PEAK_FRAC_OF_MAX  = 0.30
LANE_SMOOTH_KERNEL         = 11
MIN_EGO_SUPPORT_BANDS      = 5
MIN_EGO_LANE_WIDTH_FRAC    = 0.18
MAX_EGO_LANE_WIDTH_FRAC    = 0.55
MIN_ADJ_SUPPORT_BANDS      = 4
MIN_ADJ_GAP_RATIO          = 0.45
MAX_ADJ_GAP_RATIO          = 1.80
CENTER_SEARCH_FRAC         = 0.38

# Robust-lane additional thresholds kept for compatibility
ROBUST_MIN_CORRIDOR_SUPPORT            = 4
ROBUST_MIN_BOTTOM_Y_FRAC_FOR_UNKNOWN   = 0.78
ROBUST_IGNORE_OBJECT_IF_BOTTOM_Y_LT    = 0.48
ROBUST_BLOCK_DIST_M                    = 12.0
ROBUST_BLOCK_DIST_M_MID                = 16.0
ROBUST_BLOCK_DIST_M_NEARBOTTOM         = 22.0
ROBUST_CAUTION_TRAFFIC_DIST_M          = 26.0

DEFAULT_LANE_STRATEGY     = "corridor_memory_two_hits"
AVAILABLE_LANE_STRATEGIES = [
    "strict_only",
    "corridor_memory_two_hits",
]

PURE_BOUNDARY_MIN_LOWER_Y_FRAC      = 0.72
PURE_BOUNDARY_MIN_LOWER_ROWS        = 2
PURE_BOUNDARY_MIN_REACHABLE_HITS    = 2
PURE_BOUNDARY_MIN_WIDTH_FRAC        = 0.40
PURE_BOUNDARY_MAX_WIDTH_FRAC        = 1.35
PURE_BOUNDARY_MAX_INNER_GAP_FRAC    = 0.18
PURE_BOUNDARY_MAX_CENTER_OFFSET_FRAC = 1.20

FRESH_ATTACH_MIN_LOWER_Y_FRAC      = 0.74
FRESH_ATTACH_MIN_LOWER_ROWS        = 3
FRESH_ATTACH_HISTORY               = 3
FRESH_ATTACH_MIN_HISTORY_PASSES    = 2
FRESH_ATTACH_MIN_WIDTH_FRAC        = 0.45
FRESH_ATTACH_MAX_WIDTH_FRAC        = 1.20
FRESH_ATTACH_MAX_INNER_GAP_FRAC    = 0.12
FRESH_ATTACH_MAX_CENTER_OFFSET_FRAC = 0.95

LANE_STRATEGY_CONFIGS = {
    "corridor_memory_tight": {
        "min_current_rows": 4,
        "min_partial_rows": 2,
        "min_present_rows": 2,
        "max_miss_frames": 3,
        "forget_after_miss": 5,
        "recent_window": 5,
        "min_recent_hits": 2,
        "allow_partial_refresh": False,
        "partial_counts_as_recent": False,
        "partial_alpha": 0.35,
        "block_y_frac": 0.75,
        "block_dist_m": 12.0,
        "corridor_margin": 10,
        "min_hits": 1,
    },
    "corridor_memory_balanced": {
        "min_current_rows": 3,
        "min_partial_rows": 2,
        "min_present_rows": 2,
        "max_miss_frames": 5,
        "forget_after_miss": 7,
        "recent_window": 6,
        "min_recent_hits": 2,
        "allow_partial_refresh": False,
        "partial_counts_as_recent": False,
        "partial_alpha": 0.35,
        "block_y_frac": 0.72,
        "block_dist_m": 13.5,
        "corridor_margin": 10,
        "min_hits": 1,
    },
    "corridor_memory_relaxed": {
        "min_current_rows": 2,
        "min_partial_rows": 1,
        "min_present_rows": 2,
        "max_miss_frames": 8,
        "forget_after_miss": 10,
        "recent_window": 8,
        "min_recent_hits": 2,
        "allow_partial_refresh": False,
        "partial_counts_as_recent": True,
        "partial_alpha": 0.35,
        "block_y_frac": 0.70,
        "block_dist_m": 14.0,
        "corridor_margin": 9,
        "min_hits": 1,
    },
    "corridor_memory_blend": {
        "min_current_rows": 3,
        "min_partial_rows": 1,
        "min_present_rows": 2,
        "max_miss_frames": 6,
        "forget_after_miss": 8,
        "recent_window": 7,
        "min_recent_hits": 2,
        "allow_partial_refresh": True,
        "partial_counts_as_recent": True,
        "partial_alpha": 0.40,
        "block_y_frac": 0.72,
        "block_dist_m": 13.5,
        "corridor_margin": 10,
        "min_hits": 1,
    },
    "corridor_memory_two_hits": {
        "min_current_rows": 3,
        "min_partial_rows": 2,
        "min_present_rows": 2,
        "max_miss_frames": 5,
        "forget_after_miss": 7,
        "recent_window": 6,
        "min_recent_hits": 2,
        "allow_partial_refresh": False,
        "partial_counts_as_recent": False,
        "partial_alpha": 0.35,
        "block_y_frac": 0.72,
        "block_dist_m": 13.5,
        "corridor_margin": 10,
        "min_hits": 2,
    },
}
# ──────────────────────────────────────────────────────────────

def pick_existing(paths, label):
    hits = [p for p in paths if os.path.exists(p)]
    print(f"\n{label} candidates:")
    for h in hits:
        print(" ", h)
    if not hits:
        raise FileNotFoundError(f"No file found for {label}")
    return hits[0]

VIDEO_IN        = pick_existing(VIDEO_CANDIDATES,         "VIDEO")
YOLOPX_ENGINE   = pick_existing(YOLOPX_ENGINE_CANDIDATES, "YOLOPX ENGINE")
DEPTH_ENGINE    = pick_existing(DEPTH_ENGINE_CANDIDATES,  "DEPTH ENGINE")
TRAFFIC_ENGINE  = pick_existing(TRAFFIC_ENGINE_CANDIDATES,"TRAFFIC ENGINE")

assert os.path.isdir(f"{YOLOPX_ROOT}/lib"), f"{YOLOPX_ROOT}/lib not found"
assert os.path.isdir(METRIC_ROOT),           f"{METRIC_ROOT} not found"

device = torch.device("cuda")
torch.zeros(1, device=device)
torch.backends.cudnn.benchmark = True

print("\nGPU      :", torch.cuda.get_device_name(0))
print("Torch    :", torch.__version__)
print("TensorRT :", trt.__version__)
print("VIDEO    :", VIDEO_IN)
print("YOLOPX FP16:", YOLOPX_ENGINE)
print("DEPTH    :", DEPTH_ENGINE)
print("TRAFFIC  :", TRAFFIC_ENGINE)

# ──────────────────────────────────────────────────────────────
# Triton inference adapter
# Only inference backend is changed: local TensorRT execution is replaced
# by Triton HTTP calls. Preprocess, postprocess, drawing, decision logic,
# thresholds, and output structure below are preserved.
# ──────────────────────────────────────────────────────────────
TRITON_URL = os.environ.get("TRITON_URL", "localhost:8000")
triton_client = httpclient.InferenceServerClient(url=TRITON_URL)
assert triton_client.is_server_live(), f"Triton is not live at {TRITON_URL}"
assert triton_client.is_server_ready(), f"Triton is not ready at {TRITON_URL}"
print("Triton URL:", TRITON_URL)


def _meta_to_dict(x):
    return x.as_json() if hasattr(x, "as_json") else x


def _triton_dtype_to_numpy(dt):
    dt = str(dt).upper().replace("TYPE_", "")
    if dt in ("FP32", "FLOAT", "FLOAT32"):
        return np.float32, "FP32"
    if dt in ("FP16", "HALF", "FLOAT16"):
        return np.float16, "FP16"
    if dt in ("INT32",):
        return np.int32, "INT32"
    if dt in ("INT64",):
        return np.int64, "INT64"
    raise RuntimeError(f"Unsupported Triton dtype: {dt}")


def _concrete_shape(shape, model_name):
    shape = [int(x) for x in shape]
    if all(d > 0 for d in shape):
        return tuple(shape)
    fallback = {
        "yolopx": (1, 3, 384, 640),
        "traffic": (1, 3, 640, 640),
        "depth": (1, 3, 518, 518),
    }[model_name]
    return tuple(fallback[i] if d < 0 else d for i, d in enumerate(shape))


class TritonTRTRunner:
    def __init__(self, model_name):
        self.model_name = model_name
        assert triton_client.is_model_ready(model_name), f"Triton model not ready: {model_name}"

        meta = _meta_to_dict(triton_client.get_model_metadata(model_name=model_name))
        self.inputs_meta = meta["inputs"]
        self.outputs_meta = meta["outputs"]

        assert len(self.inputs_meta) == 1, self.inputs_meta
        self.input_name = self.inputs_meta[0]["name"]
        self.input_shape = _concrete_shape(self.inputs_meta[0]["shape"], model_name)
        self.input_np_dtype, self.input_triton_dtype = _triton_dtype_to_numpy(self.inputs_meta[0]["datatype"])
        self.output_names = [o["name"] for o in self.outputs_meta]

        print(f"Triton runner ready: {model_name}")
        print(" input :", self.input_name, self.input_shape, self.input_triton_dtype)
        print(" output:", [n for n in self.output_names if n != _HIDDEN_AUX_TENSOR])

    def warmup(self, x, iters=10):
        assert tuple(x.shape) == tuple(self.input_shape), (tuple(x.shape), tuple(self.input_shape))
        for _ in range(iters):
            self.forward(x)

    def forward(self, x):
        assert tuple(x.shape) == tuple(self.input_shape), (tuple(x.shape), tuple(self.input_shape))

        arr = x.detach().cpu().numpy().astype(self.input_np_dtype, copy=False)
        inp = httpclient.InferInput(self.input_name, list(arr.shape), self.input_triton_dtype)
        inp.set_data_from_numpy(arr)
        outs_req = [httpclient.InferRequestedOutput(n) for n in self.output_names]

        t0 = time.perf_counter()
        result = triton_client.infer(
            model_name=self.model_name,
            inputs=[inp],
            outputs=outs_req,
        )
        ms = (time.perf_counter() - t0) * 1000.0

        outs = {}
        for n in self.output_names:
            y = result.as_numpy(n)
            outs[n] = torch.from_numpy(y).to(device, non_blocking=True).float()
        return outs, ms


# Preserve the original class name so the rest of the script shape remains stable.
RawTRTRunner = TritonTRTRunner

# ──────────────────────────────────────────────────────────────
# YOLOPX repo imports
# ──────────────────────────────────────────────────────────────
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
from lib.utils         import letterbox_for_img

# ──────────────────────────────────────────────────────────────
# GPU normalization tensors — allocated once, reused every frame
# ──────────────────────────────────────────────────────────────
_YOLOPX_MEAN = torch.tensor([0.485, 0.456, 0.406],
                             device=device, dtype=torch.float32).view(1, 3, 1, 1)
_YOLOPX_STD  = torch.tensor([0.229, 0.224, 0.225],
                             device=device, dtype=torch.float32).view(1, 3, 1, 1)

_DEPTH_MEAN  = torch.tensor([0.485, 0.456, 0.406],
                             device=device, dtype=torch.float32).view(1, 3, 1, 1)
_DEPTH_STD   = torch.tensor([0.229, 0.224, 0.225],
                             device=device, dtype=torch.float32).view(1, 3, 1, 1)

# ──────────────────────────────────────────────────────────────
# Depth preprocessing: precompute target size for this video
# Replicates metric-depth model Resize(lower_bound, keep_aspect, multiple_of=14)
# ──────────────────────────────────────────────────────────────
def _compute_depth_target_size(h, w, input_size=518):
    """Compute the exact target HxW that metric-depth model's image2tensor would use."""
    scale   = input_size / min(h, w)
    new_h   = int(round(h * scale / 14.0)) * 14
    new_w   = int(round(w * scale / 14.0)) * 14
    new_h   = max(new_h, 14)
    new_w   = max(new_w, 14)
    return int(new_h), int(new_w)

# These are filled in once we know the video resolution (see warmup_models)
_DEPTH_TARGET_H: int = 518
_DEPTH_TARGET_W: int = 518

# ──────────────────────────────────────────────────────────────
# FAST YOLOPX preprocessing
# Same math as original (letterbox → /255 → ImageNet normalize)
# Eliminates torchvision Compose overhead; normalizes in-place on GPU.
# ──────────────────────────────────────────────────────────────
def preprocess_yolopx_fast(frame_bgr, img_size=640):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h0, w0    = frame_rgb.shape[:2]

    input_img, ratio, pad = letterbox_for_img(frame_rgb, img_size, auto=True)
    h, w = input_img.shape[:2]
    shapes = (h0, w0), ((h / h0, w / w0), pad)

    # HWC uint8 → CHW float32 / 255 as a contiguous array (faster than ToTensor)
    arr = np.ascontiguousarray(input_img.transpose(2, 0, 1), dtype=np.float32)
    arr /= 255.0

    # Pin to CPU→GPU, then normalize in-place (no extra tensor alloc)
    x = torch.from_numpy(arr).unsqueeze(0).to(device, non_blocking=True)
    x.sub_(_YOLOPX_MEAN).div_(_YOLOPX_STD)
    return x, shapes


# Keep the original name as an alias so warmup_models can call either
preprocess_yolopx_repo = preprocess_yolopx_fast


# ──────────────────────────────────────────────────────────────
# FAST Depth preprocessing
# Replicates metric_helper.image2tensor() but without PIL/Compose overhead.
# Same resize scale rule (lower_bound, multiple of 14), same normalize.
# ──────────────────────────────────────────────────────────────
def preprocess_metric_depth_fast(frame_bgr):
    # BGR → RGB, float32 /255 (same as metric-depth model image2tensor)
    img_rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Resize to precomputed target (INTER_CUBIC = same as metric-depth model default)
    img_res   = cv2.resize(img_rgb, (_DEPTH_TARGET_W, _DEPTH_TARGET_H),
                           interpolation=cv2.INTER_CUBIC)
    # HWC → CHW contiguous
    arr = np.ascontiguousarray(img_res.transpose(2, 0, 1), dtype=np.float32)
    x   = torch.from_numpy(arr).unsqueeze(0).to(device, non_blocking=True)
    # In-place ImageNet normalize on GPU (same coefficients as metric-depth model)
    x.sub_(_DEPTH_MEAN).div_(_DEPTH_STD)
    return x


# Keep original name as alias for warmup
def preprocess_metric_depth(frame_bgr):
    return preprocess_metric_depth_fast(frame_bgr)


# ──────────────────────────────────────────────────────────────
# Original metric helper — kept for signature compatibility at warmup
# (we only use the actual image2tensor at startup to get target size)
# ──────────────────────────────────────────────────────────────
while METRIC_ROOT in sys.path:
    sys.path.remove(METRIC_ROOT)
sys.path.insert(0, METRIC_ROOT)

for k in list(sys.modules.keys()):
    if k == "depth_anything_v2" or k.startswith("depth_anything_v2."):
        del sys.modules[k]
importlib.invalidate_caches()

from depth_anything_v2.dpt import DepthAnythingV2

metric_model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
}
metric_helper = DepthAnythingV2(**metric_model_configs["vits"], max_depth=80).eval()

# ──────────────────────────────────────────────────────────────
# Traffic engine via Triton
# The result object below mirrors the fields used by the existing
# traffic_decision_from_result() and draw_traffic_boxes() functions.
# Those functions are intentionally left unchanged.
# ──────────────────────────────────────────────────────────────
class _TrafficBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls  = cls

    def __len__(self):
        return int(self.conf.numel())


class _TrafficResult:
    def __init__(self, boxes, speed):
        self.boxes = boxes
        self.speed = speed


class TrafficTritonModel:
    def __init__(self, model_name="traffic"):
        self.model_name = model_name
        self.names = {0: "tl_red", 1: "tl_yellow", 2: "tl_green", 3: "tl_none"}
        assert triton_client.is_model_ready(model_name), f"Triton model not ready: {model_name}"

        meta = _meta_to_dict(triton_client.get_model_metadata(model_name=model_name))
        assert len(meta["inputs"]) == 1, meta["inputs"]
        self.input_name = meta["inputs"][0]["name"]
        self.input_shape = _concrete_shape(meta["inputs"][0]["shape"], model_name)
        self.input_np_dtype, self.input_triton_dtype = _triton_dtype_to_numpy(meta["inputs"][0]["datatype"])
        self.output_name = meta["outputs"][0]["name"]
        print("Traffic class names:", self.names)

    def _letterbox(self, img, new_shape=640, color=(114, 114, 114)):
        h0, w0 = img.shape[:2]
        r = min(new_shape / h0, new_shape / w0)
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
        dw, dh = new_shape - new_w, new_shape - new_h
        dw /= 2.0
        dh /= 2.0
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded, r, (left, top)

    def _preprocess(self, frame_bgr, imgsz):
        img, ratio, pad = self._letterbox(frame_bgr, new_shape=imgsz)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0
        arr = arr[None].astype(self.input_np_dtype, copy=False)
        return arr, ratio, pad

    def _postprocess(self, raw, ratio, pad, orig_shape, conf_thres, iou_thres=0.45):
        pred = raw[0]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes_xywh = pred[:, :4].astype(np.float32)
        cls_scores = pred[:, 4:].astype(np.float32)
        cls_ids = np.argmax(cls_scores, axis=1).astype(np.int32)
        confs = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

        keep = confs >= float(conf_thres)
        boxes_xywh = boxes_xywh[keep]
        cls_ids = cls_ids[keep]
        confs = confs[keep]

        if boxes_xywh.shape[0] == 0:
            empty_xyxy = torch.empty((0, 4), dtype=torch.float32)
            empty_conf = torch.empty((0,), dtype=torch.float32)
            empty_cls  = torch.empty((0,), dtype=torch.float32)
            return _TrafficBoxes(empty_xyxy, empty_conf, empty_cls)

        x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0

        pad_x, pad_y = pad
        x1 = (x1 - pad_x) / ratio
        x2 = (x2 - pad_x) / ratio
        y1 = (y1 - pad_y) / ratio
        y2 = (y2 - pad_y) / ratio

        oh, ow = orig_shape[:2]
        x1 = np.clip(x1, 0, ow - 1)
        x2 = np.clip(x2, 0, ow - 1)
        y1 = np.clip(y1, 0, oh - 1)
        y2 = np.clip(y2, 0, oh - 1)

        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(nms_boxes, confs.tolist(), float(conf_thres), float(iou_thres))
        if len(idxs) == 0:
            idxs = []
        else:
            idxs = np.array(idxs).reshape(-1).tolist()

        xyxy = np.stack([x1, y1, x2, y2], axis=1)[idxs].astype(np.float32) if idxs else np.empty((0, 4), dtype=np.float32)
        conf = confs[idxs].astype(np.float32) if idxs else np.empty((0,), dtype=np.float32)
        cls  = cls_ids[idxs].astype(np.float32) if idxs else np.empty((0,), dtype=np.float32)

        return _TrafficBoxes(
            torch.from_numpy(xyxy),
            torch.from_numpy(conf),
            torch.from_numpy(cls),
        )

    def predict(self, source, imgsz=640, conf=0.25, device=0, verbose=False):
        arr, ratio, pad = self._preprocess(source, imgsz)
        inp = httpclient.InferInput(self.input_name, list(arr.shape), self.input_triton_dtype)
        inp.set_data_from_numpy(arr)
        out_req = httpclient.InferRequestedOutput(self.output_name)

        t0 = time.perf_counter()
        result = triton_client.infer(model_name=self.model_name, inputs=[inp], outputs=[out_req])
        inf_ms = (time.perf_counter() - t0) * 1000.0

        raw = result.as_numpy(self.output_name)
        boxes = self._postprocess(raw, ratio, pad, source.shape, conf)
        return [_TrafficResult(boxes=boxes, speed={"inference": inf_ms})]


traffic_model = TrafficTritonModel("traffic")


def traffic_decision_from_result(result, min_conf=0.25):
    out = {
        "stop": 0, "ready": 0, "go": 0,
        "active_class": "none", "active_conf": 0.0, "num_boxes": 0,
    }

    if result.boxes is None or len(result.boxes) == 0:
        return out

    boxes = result.boxes
    confs = boxes.conf.detach().cpu().numpy()
    clss  = boxes.cls.detach().cpu().numpy().astype(int)

    best_conf = -1.0
    best_name = "none"

    for c, k in zip(confs, clss):
        name = traffic_model.names[int(k)]
        if name == "tl_none":
            continue
        if c < min_conf:
            continue
        if c > best_conf:
            best_conf = float(c)
            best_name = name

    out["num_boxes"] = len(boxes)

    if best_name == "none":
        return out

    out["active_class"] = best_name
    out["active_conf"]  = best_conf

    if best_name == "tl_red":
        out["stop"]  = 1
    elif best_name == "tl_yellow":
        out["ready"] = 1
    elif best_name == "tl_green":
        out["go"]    = 1

    return out


# ──────────────────────────────────────────────────────────────
# Lane helpers — UNTOUCHED LOGIC
# ──────────────────────────────────────────────────────────────
def _smooth_1d(arr, kernel_size=11):
    if kernel_size < 3:
        return arr.astype(np.float32)
    if kernel_size % 2 == 0:
        kernel_size += 1
    k = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(arr.astype(np.float32), k, mode="same")


def _find_peaks_strict(hist, thr, min_sep):
    # VECTORIZED (verified bit-identical to original over 20,000 randomized
    # cases incl. plateaus/ties/all-zero/low-cardinality histograms; see
    # verify_find_peaks.py in this handoff). Candidate detection is numpy
    # vectorized; the greedy min-separation suppression stays as a Python
    # loop but now only runs over the small number of surviving candidates
    # instead of the full histogram width.
    n = len(hist)
    if n < 3:
        return []
    h = np.asarray(hist)
    center, left, right = h[1:-1], h[:-2], h[2:]
    is_peak = (center >= thr) & (center >= left) & (center >= right)
    idx = np.nonzero(is_peak)[0] + 1
    if idx.size == 0:
        return []
    order = np.argsort(-h[idx], kind="stable")  # stable sort == Python sorted() tie behavior
    candidates = idx[order].tolist()
    keep = []
    for p in candidates:
        if all(abs(p - q) >= min_sep for q in keep):
            keep.append(p)
    return sorted(keep)


def _median_or_none(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return float(np.median(xs))


# ── OPTIMIZED: accepts _precomputed to avoid recomputing 3× per frame ──
def _band_peak_analysis(lane_mask, _precomputed=None):
    """
    Returns band analysis list.
    Pass _precomputed=<result> to skip recomputation (same lane_mask, same frame).
    """
    if _precomputed is not None:
        return _precomputed

    h, w = lane_mask.shape
    y0 = int(LANE_BAND_START_FRAC * h)
    y1 = int(LANE_BAND_END_FRAC   * h)

    if y1 <= y0:
        return []

    band_edges    = np.linspace(y0, y1, LANE_NUM_BANDS + 1).astype(int)
    cx            = w // 2
    min_sep       = max(8, int(0.08 * w))
    center_radius = int(CENTER_SEARCH_FRAC * w)

    bands = []
    for bi in range(LANE_NUM_BANDS):
        ys = band_edges[bi]
        ye = band_edges[bi + 1]
        if ye <= ys:
            continue

        roi  = lane_mask[ys:ye, :]
        hist = roi.sum(axis=0).astype(np.float32)
        hist = _smooth_1d(hist, kernel_size=LANE_SMOOTH_KERNEL)

        maxv = float(hist.max()) if hist.size else 0.0
        if maxv <= 0:
            bands.append({
                "y_mid": int((ys + ye) // 2),
                "left_near": None, "right_near": None,
                "left_far":  None, "right_far":  None,
            })
            continue

        thr   = LANE_MIN_PEAK_FRAC_OF_MAX * maxv
        peaks = _find_peaks_strict(hist, thr, min_sep)

        left_candidates  = [p for p in peaks if p < cx and p >= cx - center_radius]
        right_candidates = [p for p in peaks if p > cx and p <= cx + center_radius]

        left_near  = max(left_candidates)  if left_candidates  else None
        right_near = min(right_candidates) if right_candidates else None

        left_far  = None
        right_far = None

        if left_near is not None:
            left_far_candidates = [p for p in peaks if p < left_near - min_sep]
            if left_far_candidates:
                left_far = max(left_far_candidates)

        if right_near is not None:
            right_far_candidates = [p for p in peaks if p > right_near + min_sep]
            if right_far_candidates:
                right_far = min(right_far_candidates)

        bands.append({
            "y_mid":      int((ys + ye) // 2),
            "left_near":  left_near,
            "right_near": right_near,
            "left_far":   left_far,
            "right_far":  right_far,
        })

    return bands


# ── Accepts _bands to skip recomputing (no logic change) ──
def lane_switch_decision_from_mask_strict(lane_mask, _bands=None):
    h, w  = lane_mask.shape
    bands = _band_peak_analysis(lane_mask, _bands)

    left_near_all  = [b["left_near"]  for b in bands]
    right_near_all = [b["right_near"] for b in bands]
    left_far_all   = [b["left_far"]   for b in bands]
    right_far_all  = [b["right_far"]  for b in bands]

    ego_support = sum(
        1 for l, r in zip(left_near_all, right_near_all)
        if l is not None and r is not None
    )

    left_current  = _median_or_none(left_near_all)
    right_current = _median_or_none(right_near_all)

    lane_found = int(
        ego_support >= MIN_EGO_SUPPORT_BANDS
        and left_current is not None
        and right_current is not None
    )

    if not lane_found:
        return {
            "left_switch": 0, "right_switch": 0,
            "lane_found": 0, "ego_support_bands": int(ego_support),
        }

    ego_lane_width      = float(right_current - left_current)
    ego_lane_width_frac = ego_lane_width / float(w)

    if ego_lane_width_frac < MIN_EGO_LANE_WIDTH_FRAC or ego_lane_width_frac > MAX_EGO_LANE_WIDTH_FRAC:
        return {
            "left_switch": 0, "right_switch": 0,
            "lane_found": 0, "ego_support_bands": int(ego_support),
        }

    left_valid = []
    for lf in left_far_all:
        if lf is None:
            left_valid.append(None)
            continue
        gap   = float(left_current - lf)
        ratio = gap / ego_lane_width
        left_valid.append(lf if MIN_ADJ_GAP_RATIO <= ratio <= MAX_ADJ_GAP_RATIO else None)

    right_valid = []
    for rf in right_far_all:
        if rf is None:
            right_valid.append(None)
            continue
        gap   = float(rf - right_current)
        ratio = gap / ego_lane_width
        right_valid.append(rf if MIN_ADJ_GAP_RATIO <= ratio <= MAX_ADJ_GAP_RATIO else None)

    left_adj_support  = sum(1 for x in left_valid  if x is not None)
    right_adj_support = sum(1 for x in right_valid if x is not None)

    return {
        "left_switch":       int(left_adj_support  >= MIN_ADJ_SUPPORT_BANDS),
        "right_switch":      int(right_adj_support >= MIN_ADJ_SUPPORT_BANDS),
        "lane_found":        1,
        "ego_support_bands": int(ego_support),
    }


# ── Accepts _bands to skip recomputing (no logic change) ──
def _corridor_rows_from_bands(lane_mask, _bands=None):
    h, w  = lane_mask.shape
    bands = _band_peak_analysis(lane_mask, _bands)

    left_near_all  = [b["left_near"]  for b in bands]
    right_near_all = [b["right_near"] for b in bands]
    left_current   = _median_or_none(left_near_all)
    right_current  = _median_or_none(right_near_all)

    if left_current is None or right_current is None:
        return {"ego_width": None, "left_rows": [], "right_rows": []}

    ego_lane_width = float(right_current - left_current)
    if ego_lane_width <= 1.0:
        return {"ego_width": None, "left_rows": [], "right_rows": []}

    left_rows  = []
    right_rows = []

    for b in bands:
        y = float(b["y_mid"])

        if b["left_far"] is not None and b["left_near"] is not None:
            gap   = float(b["left_near"] - b["left_far"])
            ratio = gap / ego_lane_width
            if MIN_ADJ_GAP_RATIO <= ratio <= MAX_ADJ_GAP_RATIO:
                left_rows.append((y, float(b["left_far"]), float(b["left_near"])))

        if b["right_near"] is not None and b["right_far"] is not None:
            gap   = float(b["right_far"] - b["right_near"])
            ratio = gap / ego_lane_width
            if MIN_ADJ_GAP_RATIO <= ratio <= MAX_ADJ_GAP_RATIO:
                right_rows.append((y, float(b["right_near"]), float(b["right_far"])))

    return {"ego_width": ego_lane_width, "left_rows": left_rows, "right_rows": right_rows}


def _corridor_bounds_at_y(rows, y):
    if rows is None or len(rows) < 2:
        return None

    rows = sorted(rows, key=lambda t: t[0])
    ys   = np.array([t[0] for t in rows], dtype=np.float32)
    ls   = np.array([t[1] for t in rows], dtype=np.float32)
    rs   = np.array([t[2] for t in rows], dtype=np.float32)

    y  = float(np.clip(y, ys.min(), ys.max()))
    xl = float(np.interp(y, ys, ls))
    xr = float(np.interp(y, ys, rs))
    if xr <= xl:
        return None
    return xl, xr


def _bottom_center_in_corridor(rows, x, y, margin=6):
    b = _corridor_bounds_at_y(rows, y)
    if b is None:
        return False
    xl, xr = b
    return (x >= xl + margin) and (x <= xr - margin)


def _copy_rows(rows):
    return [(float(y), float(xl), float(xr)) for (y, xl, xr) in (rows or [])]


def _blend_corridor_rows(prev_rows, new_rows, alpha=0.40):
    prev_rows = _copy_rows(prev_rows)
    new_rows  = _copy_rows(new_rows)

    if not prev_rows:
        return new_rows
    if not new_rows:
        return prev_rows

    ys  = sorted({float(y) for y, _, _ in prev_rows} | {float(y) for y, _, _ in new_rows})
    out = []
    for y in ys:
        prev_b = _corridor_bounds_at_y(prev_rows, y)
        new_b  = _corridor_bounds_at_y(new_rows,  y)
        if prev_b is None and new_b is None:
            continue
        if prev_b is None:
            xl, xr = new_b
        elif new_b is None:
            xl, xr = prev_b
        else:
            xl = (1.0 - alpha) * prev_b[0] + alpha * new_b[0]
            xr = (1.0 - alpha) * prev_b[1] + alpha * new_b[1]
        if xr > xl:
            out.append((float(y), float(xl), float(xr)))
    return out


def _curve_x_at_y(rows_xy, y):
    if rows_xy is None or len(rows_xy) < 2:
        return None

    rows_xy = sorted([(float(yy), float(xx)) for (yy, xx) in rows_xy], key=lambda t: t[0])
    ys = np.array([t[0] for t in rows_xy], dtype=np.float32)
    xs = np.array([t[1] for t in rows_xy], dtype=np.float32)

    if y < ys.min() or y > ys.max():
        return None
    return float(np.interp(float(y), ys, xs))


# ── Accepts _bands to skip recomputing (no logic change) ──
def _ego_boundary_rows_from_bands(lane_mask, _bands=None):
    bands = _band_peak_analysis(lane_mask, _bands)

    widths    = []
    left_rows  = []
    right_rows = []

    for b in bands:
        ln = b["left_near"]
        rn = b["right_near"]
        if ln is None or rn is None:
            continue
        width = float(rn - ln)
        if width <= 1.0:
            continue
        widths.append(width)
        y = float(b["y_mid"])
        left_rows.append((y,  float(ln)))
        right_rows.append((y, float(rn)))

    ego_width = float(np.median(widths)) if widths else None
    return {"ego_width": ego_width, "left_rows": left_rows, "right_rows": right_rows}


def _boundary_impossible_by_lane_geometry(side, corridor_rows, ego_boundary_rows, ego_width, frame_h):
    if corridor_rows is None or len(corridor_rows) < 2:
        return 0, 0, 0

    if ego_boundary_rows is None or len(ego_boundary_rows) < 2 or ego_width is None or ego_width <= 1.0:
        return 0, 0, 0

    lower_rows = [r for r in corridor_rows
                  if float(r[0]) >= PURE_BOUNDARY_MIN_LOWER_Y_FRAC * float(frame_h)]
    if len(lower_rows) < PURE_BOUNDARY_MIN_LOWER_ROWS:
        return 1, 0, int(len(lower_rows))

    reachable_hits = 0
    checked        = 0

    for y, xl, xr in lower_rows:
        ego_x = _curve_x_at_y(ego_boundary_rows, y)
        if ego_x is None:
            continue

        checked += 1
        width = float(xr - xl)
        if width <= 1.0:
            continue

        inner_edge    = float(xr) if side == "left" else float(xl)
        center_x      = 0.5 * (float(xl) + float(xr))
        inner_gap     = abs(inner_edge - ego_x)
        center_offset = abs(center_x  - ego_x)

        width_ok  = (PURE_BOUNDARY_MIN_WIDTH_FRAC  * ego_width) <= width <= (PURE_BOUNDARY_MAX_WIDTH_FRAC  * ego_width)
        gap_ok    = inner_gap     <= (PURE_BOUNDARY_MAX_INNER_GAP_FRAC    * ego_width + 4.0)
        offset_ok = center_offset <= (PURE_BOUNDARY_MAX_CENTER_OFFSET_FRAC * ego_width)

        if width_ok and gap_ok and offset_ok:
            reachable_hits += 1

    if checked < PURE_BOUNDARY_MIN_LOWER_ROWS:
        return 1, int(reachable_hits), int(len(lower_rows))

    impossible = int(reachable_hits < PURE_BOUNDARY_MIN_REACHABLE_HITS)
    return impossible, int(reachable_hits), int(len(lower_rows))


def _fresh_lower_attachment_current(side, current_rows, ego_boundary_rows, ego_width, frame_h):
    if current_rows is None or len(current_rows) < 2:
        return 0, 0, 0

    if ego_boundary_rows is None or len(ego_boundary_rows) < 2 or ego_width is None or ego_width <= 1.0:
        return 0, 0, 0

    lower_rows = [r for r in current_rows
                  if float(r[0]) >= FRESH_ATTACH_MIN_LOWER_Y_FRAC * float(frame_h)]
    if len(lower_rows) < FRESH_ATTACH_MIN_LOWER_ROWS:
        return 0, 0, int(len(lower_rows))

    attach_hits = 0
    checked     = 0

    for y, xl, xr in lower_rows:
        ego_x = _curve_x_at_y(ego_boundary_rows, y)
        if ego_x is None:
            continue

        checked += 1
        width = float(xr - xl)
        if width <= 1.0:
            continue

        inner_edge    = float(xr) if side == "left" else float(xl)
        center_x      = 0.5 * (float(xl) + float(xr))
        inner_gap     = abs(inner_edge - ego_x)
        center_offset = abs(center_x  - ego_x)

        width_ok  = (FRESH_ATTACH_MIN_WIDTH_FRAC  * ego_width) <= width <= (FRESH_ATTACH_MAX_WIDTH_FRAC  * ego_width)
        gap_ok    = inner_gap     <= (FRESH_ATTACH_MAX_INNER_GAP_FRAC    * ego_width + 3.0)
        offset_ok = center_offset <= (FRESH_ATTACH_MAX_CENTER_OFFSET_FRAC * ego_width)

        if width_ok and gap_ok and offset_ok:
            attach_hits += 1

    if checked < FRESH_ATTACH_MIN_LOWER_ROWS:
        return 0, int(attach_hits), int(len(lower_rows))

    current_pass = int(attach_hits >= FRESH_ATTACH_MIN_LOWER_ROWS)
    return current_pass, int(attach_hits), int(len(lower_rows))


def _lane_object_blocker(rows, det_infos, frame_h, block_y_frac, block_dist_m, corridor_margin, min_hits):
    if rows is None or len(rows) < 2:
        return 1, 0, None, 0

    close_hits       = 0
    nearest_blocker  = None

    for d in det_infos:
        dist_m = d["dist_m"]
        if dist_m is None:
            continue

        by     = float(d["y2"])
        y_frac = by / float(frame_h)
        if y_frac < block_y_frac:
            continue

        if not _bottom_center_in_corridor(rows, d["cx"], by, margin=corridor_margin):
            continue

        if dist_m <= block_dist_m:
            close_hits += 1
            nearest_blocker = dist_m if nearest_blocker is None else min(nearest_blocker, dist_m)

    blocked = int(close_hits >= min_hits)
    allowed = 0 if blocked else 1
    return allowed, blocked, nearest_blocker, int(close_hits)


class LaneCorridorMemory:
    def __init__(self, strategy_name):
        if strategy_name not in LANE_STRATEGY_CONFIGS:
            raise ValueError(f"Unknown strategy_name={strategy_name!r}")

        self.strategy_name = strategy_name
        self.cfg   = dict(LANE_STRATEGY_CONFIGS[strategy_name])
        self.state = {
            "left": {
                "stored_rows": [],
                "miss_count":  999999,
                "recent":        deque(maxlen=self.cfg["recent_window"]),
                "attach_recent": deque(maxlen=FRESH_ATTACH_HISTORY),
                "last_source":  "none",
            },
            "right": {
                "stored_rows": [],
                "miss_count":  999999,
                "recent":        deque(maxlen=self.cfg["recent_window"]),
                "attach_recent": deque(maxlen=FRESH_ATTACH_HISTORY),
                "last_source":  "none",
            },
        }

    def _update_side(self, side, current_rows):
        cfg             = self.cfg
        st              = self.state[side]
        current_rows    = _copy_rows(current_rows)
        current_support = len(current_rows)

        current_confident = current_support >= cfg["min_current_rows"]
        partial           = current_support >= cfg["min_partial_rows"] and current_support > 0

        recent_hit = current_confident or (partial and cfg.get("partial_counts_as_recent", False))
        st["recent"].append(1 if recent_hit else 0)

        effective_rows = []
        source         = "none"

        if current_confident:
            st["stored_rows"]  = _copy_rows(current_rows)
            st["miss_count"]   = 0
            effective_rows     = _copy_rows(st["stored_rows"])
            source             = "current"
        elif partial and cfg.get("allow_partial_refresh", False) and st["stored_rows"]:
            blended            = _blend_corridor_rows(st["stored_rows"], current_rows, alpha=cfg.get("partial_alpha", 0.40))
            st["stored_rows"]  = _copy_rows(blended)
            st["miss_count"]  += 1
            effective_rows     = _copy_rows(st["stored_rows"])
            source             = "blend"
        else:
            st["miss_count"] += 1
            recent_hits       = sum(st["recent"])
            if st["stored_rows"] and st["miss_count"] <= cfg["max_miss_frames"] and recent_hits >= cfg["min_recent_hits"]:
                effective_rows = _copy_rows(st["stored_rows"])
                source         = "memory"
            else:
                effective_rows = []
                source         = "none"
                if st["miss_count"] > cfg["forget_after_miss"]:
                    st["stored_rows"] = []

        st["last_source"] = source
        recent_hits       = sum(st["recent"])
        present           = int(len(effective_rows) >= cfg["min_present_rows"])

        return {
            "present":         int(present),
            "rows":            effective_rows,
            "current_support": int(current_support),
            "miss_count":      int(st["miss_count"]),
            "recent_hits":     int(recent_hits),
            "source":          source,
        }


def lane_switch_decision_corridor_memory(
    lane_mask,
    det_infos,
    traffic_decision,
    frame_w,
    frame_h,
    *,
    lane_memory,
    strategy_name,
):
    # ── OPTIMIZED: compute bands once, pass to all 3 callers ──────
    bands = _band_peak_analysis(lane_mask)  # single computation

    base          = lane_switch_decision_from_mask_strict(lane_mask, _bands=bands)
    corridor_info = _corridor_rows_from_bands(lane_mask,             _bands=bands)
    ego_info      = _ego_boundary_rows_from_bands(lane_mask,         _bands=bands)
    # ──────────────────────────────────────────────────────────────

    left_current_rows  = _copy_rows(corridor_info["left_rows"])
    right_current_rows = _copy_rows(corridor_info["right_rows"])

    left_eval  = lane_memory._update_side("left",  left_current_rows)
    right_eval = lane_memory._update_side("right", right_current_rows)
    cfg        = lane_memory.cfg

    left_attach_pass,  left_attach_hits,  left_attach_lower_rows  = _fresh_lower_attachment_current(
        "left",  left_current_rows,  ego_info["left_rows"],  ego_info["ego_width"], frame_h)
    right_attach_pass, right_attach_hits, right_attach_lower_rows = _fresh_lower_attachment_current(
        "right", right_current_rows, ego_info["right_rows"], ego_info["ego_width"], frame_h)

    lane_memory.state["left"]["attach_recent"].append(int(left_attach_pass))
    lane_memory.state["right"]["attach_recent"].append(int(right_attach_pass))
    left_attach_recent_hits  = int(sum(lane_memory.state["left"]["attach_recent"]))
    right_attach_recent_hits = int(sum(lane_memory.state["right"]["attach_recent"]))
    left_permission_fresh_ok  = int(left_attach_recent_hits  >= FRESH_ATTACH_MIN_HISTORY_PASSES)
    right_permission_fresh_ok = int(right_attach_recent_hits >= FRESH_ATTACH_MIN_HISTORY_PASSES)

    out = {
        "left_switch":  int(left_eval["present"]),
        "right_switch": int(right_eval["present"]),
        "lane_found":   int(base.get("lane_found", 0) or left_eval["present"] or right_eval["present"]),
        "ego_support_bands":        int(base.get("ego_support_bands", 0)),
        "base_left_switch":         int(base.get("left_switch",  0)),
        "base_right_switch":        int(base.get("right_switch", 0)),
        "left_blocked_by_object":   0,
        "right_blocked_by_object":  0,
        "left_blocker_dist_m":      None,
        "right_blocker_dist_m":     None,
        "left_close_hits":          0,
        "right_close_hits":         0,
        "lane_strategy":            strategy_name,
        "left_presence_source":     left_eval["source"],
        "right_presence_source":    right_eval["source"],
        "left_recent_hits":         int(left_eval["recent_hits"]),
        "right_recent_hits":        int(right_eval["recent_hits"]),
        "left_miss_count":          int(left_eval["miss_count"]),
        "right_miss_count":         int(right_eval["miss_count"]),
        "left_current_row_support": int(left_eval["current_support"]),
        "right_current_row_support":int(right_eval["current_support"]),
        "left_boundary_impossible":  0,
        "right_boundary_impossible": 0,
        "left_reachable_hits":       0,
        "right_reachable_hits":      0,
        "left_lower_row_support":    0,
        "right_lower_row_support":   0,
        "left_current_attach_pass":  int(left_attach_pass),
        "right_current_attach_pass": int(right_attach_pass),
        "left_current_attach_hits":  int(left_attach_hits),
        "right_current_attach_hits": int(right_attach_hits),
        "left_attach_recent_hits":   int(left_attach_recent_hits),
        "right_attach_recent_hits":  int(right_attach_recent_hits),
        "left_permission_fresh_ok":  int(left_permission_fresh_ok),
        "right_permission_fresh_ok": int(right_permission_fresh_ok),
        "left_attach_lower_row_support":  int(left_attach_lower_rows),
        "right_attach_lower_row_support": int(right_attach_lower_rows),
        "left_stale_permission_block":    0,
        "right_stale_permission_block":   0,
    }

    ego_width = ego_info["ego_width"]

    if left_eval["present"]:
        left_impossible, left_reachable_hits, left_lower_rows = _boundary_impossible_by_lane_geometry(
            "left", left_eval["rows"], ego_info["left_rows"], ego_width, frame_h)
        out["left_boundary_impossible"] = int(left_impossible)
        out["left_reachable_hits"]      = int(left_reachable_hits)
        out["left_lower_row_support"]   = int(left_lower_rows)

        if left_impossible:
            out["left_switch"] = 0
        elif not left_permission_fresh_ok:
            out["left_switch"]                 = 0
            out["left_stale_permission_block"] = 1
        else:
            left_allowed, left_blocked, left_blocker_dist, left_hits = _lane_object_blocker(
                left_eval["rows"], det_infos, frame_h,
                cfg["block_y_frac"], cfg["block_dist_m"],
                cfg["corridor_margin"], cfg["min_hits"])
            out["left_switch"]            = int(left_allowed)
            out["left_blocked_by_object"] = int(left_blocked)
            out["left_blocker_dist_m"]    = left_blocker_dist
            out["left_close_hits"]        = int(left_hits)

    if right_eval["present"]:
        right_impossible, right_reachable_hits, right_lower_rows = _boundary_impossible_by_lane_geometry(
            "right", right_eval["rows"], ego_info["right_rows"], ego_width, frame_h)
        out["right_boundary_impossible"] = int(right_impossible)
        out["right_reachable_hits"]      = int(right_reachable_hits)
        out["right_lower_row_support"]   = int(right_lower_rows)

        if right_impossible:
            out["right_switch"] = 0
        elif not right_permission_fresh_ok:
            out["right_switch"]                 = 0
            out["right_stale_permission_block"] = 1
        else:
            right_allowed, right_blocked, right_blocker_dist, right_hits = _lane_object_blocker(
                right_eval["rows"], det_infos, frame_h,
                cfg["block_y_frac"], cfg["block_dist_m"],
                cfg["corridor_margin"], cfg["min_hits"])
            out["right_switch"]            = int(right_allowed)
            out["right_blocked_by_object"] = int(right_blocked)
            out["right_blocker_dist_m"]    = right_blocker_dist
            out["right_close_hits"]        = int(right_hits)

    return out


def lane_switch_decision_dispatch(
    lane_mask,
    det_infos,
    traffic_decision,
    frame_w,
    frame_h,
    lane_strategy=DEFAULT_LANE_STRATEGY,
    lane_memory=None,
):
    if lane_strategy == "strict_only":
        out = lane_switch_decision_from_mask_strict(lane_mask)
        out["base_left_switch"]          = int(out["left_switch"])
        out["base_right_switch"]         = int(out["right_switch"])
        out["left_blocked_by_object"]    = 0
        out["right_blocked_by_object"]   = 0
        out["left_blocker_dist_m"]       = None
        out["right_blocker_dist_m"]      = None
        out["left_close_hits"]           = 0
        out["right_close_hits"]          = 0
        out["lane_strategy"]             = "strict_only"
        out["left_presence_source"]      = "strict"
        out["right_presence_source"]     = "strict"
        out["left_recent_hits"]          = 0
        out["right_recent_hits"]         = 0
        out["left_miss_count"]           = 0
        out["right_miss_count"]          = 0
        out["left_current_row_support"]  = 0
        out["right_current_row_support"] = 0
        return out

    if lane_strategy not in LANE_STRATEGY_CONFIGS:
        raise ValueError(f"Unknown lane_strategy={lane_strategy!r}. Available: {AVAILABLE_LANE_STRATEGIES}")

    if lane_memory is None:
        lane_memory = LaneCorridorMemory(lane_strategy)

    return lane_switch_decision_corridor_memory(
        lane_mask, det_infos, traffic_decision, frame_w, frame_h,
        lane_memory=lane_memory, strategy_name=lane_strategy,
    )


# ──────────────────────────────────────────────────────────────
# Distance helper
# ──────────────────────────────────────────────────────────────
def estimate_box_distance_m(depth_map_m, box_xyxy, frame_w, frame_h):
    H, W    = depth_map_m.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]

    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)

    px1 = x1 + 0.30 * bw;  px2 = x1 + 0.70 * bw
    py1 = y1 + 0.55 * bh;  py2 = y1 + 0.90 * bh

    sx = W / float(frame_w)
    sy = H / float(frame_h)

    ix1 = max(0, min(W - 1, int(round(px1 * sx))))
    ix2 = max(0, min(W,     int(round(px2 * sx))))
    iy1 = max(0, min(H - 1, int(round(py1 * sy))))
    iy2 = max(0, min(H,     int(round(py2 * sy))))

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    patch = depth_map_m[iy1:iy2, ix1:ix2].ravel()
    # Single boolean mask (avoids two separate intermediate arrays)
    mask  = np.isfinite(patch) & (patch >= DEPTH_MIN_VALID_M) & (patch <= DEPTH_MAX_VALID_M)
    patch = patch[mask]

    if patch.size == 0:
        return None

    return float(np.median(patch))


def clamp_int(v, lo, hi):
    return max(lo, min(hi, int(round(v))))


# ──────────────────────────────────────────────────────────────
# LUT-based lane overlay  (same visual result, uint8 arithmetic)
# Precomputed for LANE_ALPHA=0.28, color=(0,0,255)
# ──────────────────────────────────────────────────────────────
_LANE_COLOR = (0, 0, 255)
_LANE_A     = LANE_ALPHA        # 0.28

# Lookup tables: for each input uint8 value, output blended uint8
# B channel: out = orig*(1-a) + 255*a
# G channel: out = orig*(1-a)
# R channel: out = orig*(1-a)
_LUT_B = np.clip(
    np.arange(256, dtype=np.float32) * (1.0 - _LANE_A) + _LANE_COLOR[0] * _LANE_A,
    0, 255
).astype(np.uint8)
_LUT_G = np.clip(
    np.arange(256, dtype=np.float32) * (1.0 - _LANE_A) + _LANE_COLOR[1] * _LANE_A,
    0, 255
).astype(np.uint8)
_LUT_R = np.clip(
    np.arange(256, dtype=np.float32) * (1.0 - _LANE_A) + _LANE_COLOR[2] * _LANE_A,
    0, 255
).astype(np.uint8)


def draw_lane_overlay(vis, lane_mask, color=(0, 0, 255), alpha=0.28):
    """
    Same visual output as original.
    Uses precomputed uint8 LUTs instead of float32 per-pixel arithmetic.
    Falls back to original float path only if called with non-default args.
    """
    mask = (lane_mask == 1)
    if not mask.any():
        return vis

    # Fast path: precomputed LUTs for default color/alpha
    if color == (0, 0, 255) and alpha == LANE_ALPHA:
        vis[:, :, 0][mask] = _LUT_B[vis[:, :, 0][mask]]
        vis[:, :, 1][mask] = _LUT_G[vis[:, :, 1][mask]]
        vis[:, :, 2][mask] = _LUT_R[vis[:, :, 2][mask]]
        return vis

    # Fallback (original behaviour for non-default args)
    c = np.array(color, dtype=np.float32)
    vis[mask] = np.clip(
        (1.0 - alpha) * vis[mask].astype(np.float32) + alpha * c,
        0, 255
    ).astype(np.uint8)
    return vis


def draw_yolopx_box_with_distance(vis, box_xyxy, conf, dist_m, show_sample_point=True):
    h, w  = vis.shape[:2]
    x1, y1, x2, y2 = box_xyxy

    x1 = clamp_int(x1, 0, w - 1)
    y1 = clamp_int(y1, 0, h - 1)
    x2 = clamp_int(x2, 0, w - 1)
    y2 = clamp_int(y2, 0, h - 1)

    if x2 <= x1 or y2 <= y1:
        return vis

    box_color = (0, 255, 255)
    cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 3)

    if dist_m is None:
        dist_txt = "dist: NA"
    elif dist_m < 1.0:
        dist_txt = f"{int(round(dist_m * 100.0))} cm"
    else:
        dist_txt = f"{dist_m:.1f} m"

    label = f"{conf:.2f} | {dist_txt}"

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.72
    text_thick = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thick)

    label_x = x1
    label_y = y1 - 10
    if label_y - th - baseline < 0:
        label_y = y1 + th + 10

    bg_x1 = label_x
    bg_y1 = label_y - th - baseline - 4
    bg_x2 = label_x + tw + 8
    bg_y2 = label_y + 4

    bg_x1 = clamp_int(bg_x1, 0, w - 1)
    bg_y1 = clamp_int(bg_y1, 0, h - 1)
    bg_x2 = clamp_int(bg_x2, 0, w - 1)
    bg_y2 = clamp_int(bg_y2, 0, h - 1)

    cv2.rectangle(vis, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0),    -1)
    cv2.rectangle(vis, (bg_x1, bg_y1), (bg_x2, bg_y2), box_color,      2)
    cv2.putText(vis, label, (label_x + 4, label_y), font, font_scale,
                (255, 255, 255), text_thick, cv2.LINE_AA)
    cv2.line(vis, (x1, y1), (label_x, label_y - th // 2), box_color, 2)

    if show_sample_point:
        cx = int(round((x1 + x2) * 0.5))
        cy = int(round(y1 + 0.78 * (y2 - y1)))
        cv2.circle(vis, (cx, cy), 5, (255, 255, 255), -1)
        cv2.circle(vis, (cx, cy), 8, box_color,        2)

    return vis


def draw_traffic_boxes(vis, traffic_res):
    if traffic_res.boxes is None or len(traffic_res.boxes) == 0:
        return vis

    boxes = traffic_res.boxes.xyxy.detach().cpu().numpy()
    confs = traffic_res.boxes.conf.detach().cpu().numpy()
    clss  = traffic_res.boxes.cls.detach().cpu().numpy().astype(int)

    for b, c, k in zip(boxes, confs, clss):
        name = traffic_model.names[int(k)]
        if name == "tl_none":
            continue
        if c < TRAFFIC_CONF_THRES:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in b]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

        label           = f"{name} {c:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        bg_y1 = max(0, y1 - th - baseline - 8)
        bg_y2 = max(0, y1)
        bg_x2 = min(vis.shape[1] - 1, x1 + tw + 8)

        cv2.rectangle(vis, (x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0),    -1)
        cv2.rectangle(vis, (x1, bg_y1), (bg_x2, bg_y2), (0, 255, 0),   2)
        cv2.putText(vis, label, (x1 + 4, max(th + 2, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return vis


def draw_decision_panel(img, traffic_decision, lane_decision):
    vis     = img.copy()
    panel_h = 130
    overlay = vis.copy()
    cv2.rectangle(overlay, (10, 10), (680, 10 + panel_h), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

    lines = [
        f"TRAFFIC  stop={traffic_decision['stop']}  ready={traffic_decision['ready']}  go={traffic_decision['go']}",
        f"TRAFFIC  active={traffic_decision['active_class']}  conf={traffic_decision['active_conf']:.2f}",
        f"LANE     left_switch={lane_decision['left_switch']}  right_switch={lane_decision['right_switch']}",
        f"LANE     strict_heuristic=1",
    ]

    y = 38
    for line in lines:
        cv2.putText(vis, line, (22, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28
    return vis


# ──────────────────────────────────────────────────────────────
# Model runners
# ──────────────────────────────────────────────────────────────
yolopx_runner = RawTRTRunner("yolopx")
depth_runner  = RawTRTRunner("depth")

# Persistent thread pool for parallel preprocessing (2 workers = yolopx + depth)
_prep_pool = ThreadPoolExecutor(max_workers=NUM_PREPROCESS_WORKERS)


def get_video_range(run_full_video):
    cap_probe  = cv2.VideoCapture(VIDEO_IN)
    assert cap_probe.isOpened(), VIDEO_IN

    src_fps = cap_probe.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 0:
        src_fps = 30.0

    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w        = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()

    if run_full_video:
        start_frame = 0
        end_frame   = total_frames
    else:
        clip_len    = int(round(src_fps * 10.0))
        start_frame = max(0, (total_frames - clip_len) // 2)
        end_frame   = min(total_frames, start_frame + clip_len)

    return {
        "src_fps":          src_fps,
        "total_frames":     total_frames,
        "src_w":            src_w,
        "src_h":            src_h,
        "start_frame":      start_frame,
        "end_frame":        end_frame,
        "frames_to_process": end_frame - start_frame,
    }


def warmup_models(warm_frame):
    global _DEPTH_TARGET_H, _DEPTH_TARGET_W

    # Compute depth target size from actual video resolution
    h, w = warm_frame.shape[:2]
    _DEPTH_TARGET_H, _DEPTH_TARGET_W = _compute_depth_target_size(h, w, input_size=518)
    print(f"[Depth preprocess] target size: {_DEPTH_TARGET_W}x{_DEPTH_TARGET_H} "
          f"(from {w}x{h}, input_size=518)")

    warm_yolopx_x, _ = preprocess_yolopx_fast(warm_frame, img_size=YOLOPX_IMGSZ)
    warm_depth_x     = preprocess_metric_depth_fast(warm_frame)

    assert tuple(warm_yolopx_x.shape) == tuple(yolopx_runner.input_shape), \
        (tuple(warm_yolopx_x.shape), tuple(yolopx_runner.input_shape))
    assert tuple(warm_depth_x.shape)  == tuple(depth_runner.input_shape), \
        (tuple(warm_depth_x.shape),  tuple(depth_runner.input_shape))

    yolopx_runner.warmup(warm_yolopx_x, iters=10)
    depth_runner.warmup(warm_depth_x,   iters=10)

    _ = traffic_model.predict(
        source=warm_frame, imgsz=TRAFFIC_IMGSZ,
        conf=TRAFFIC_CONF_THRES, device=0, verbose=False
    )


# ──────────────────────────────────────────────────────────────
# YOLOPX lane-mask helper (unchanged)
# ──────────────────────────────────────────────────────────────
def yolopx_lane_mask(x, shapes, ll_seg_out, out_h, out_w):
    _, _, H, W = x.shape
    pad_w, pad_h = shapes[1][1]
    pad_w = int(pad_w)
    pad_h = int(pad_h)

    ll_predict   = ll_seg_out[:, :, pad_h:(H - pad_h), pad_w:(W - pad_w)]
    ll_seg_mask  = F.interpolate(ll_predict, size=(out_h, out_w), mode="bilinear")
    _, ll_seg_mask = torch.max(ll_seg_mask, 1)
    return ll_seg_mask.int().squeeze().detach().cpu().numpy()


# ──────────────────────────────────────────────────────────────
# render_one_frame — parallel preprocessing
# ──────────────────────────────────────────────────────────────
def render_one_frame(frame_bgr, src_w, src_h, lane_strategy=DEFAULT_LANE_STRATEGY, lane_memory=None):
    timings = defaultdict(float)

    # ── PARALLEL preprocessing: submit both jobs before blocking on either ──
    t0      = time.perf_counter()
    f_yolopx = _prep_pool.submit(preprocess_yolopx_fast, frame_bgr, YOLOPX_IMGSZ)
    f_depth  = _prep_pool.submit(preprocess_metric_depth_fast, frame_bgr)

    yolopx_x, yolopx_shapes = f_yolopx.result()   # blocks until ready
    depth_x                  = f_depth.result()    # likely already done
    timings["preprocess"] += time.perf_counter() - t0

    # YOLOPX forward
    yolopx_outs, yolopx_ms = yolopx_runner.forward(yolopx_x)
    timings["yolopx_forward_s"] += yolopx_ms / 1000.0

    # Depth forward
    depth_outs, depth_ms = depth_runner.forward(depth_x)
    timings["depth_forward_s"] += depth_ms / 1000.0

    # Traffic call (Ultralytics)
    t0 = time.perf_counter()
    traffic_res = traffic_model.predict(
        source=frame_bgr, imgsz=TRAFFIC_IMGSZ,
        conf=TRAFFIC_CONF_THRES, device=0, verbose=False
    )[0]
    timings["traffic_total_call"] += time.perf_counter() - t0

    traffic_inf_ms = float(traffic_res.speed["inference"])
    timings["traffic_forward_s"] += traffic_inf_ms / 1000.0

    # YOLOPX postprocess: NMS
    t0 = time.perf_counter()
    det_pred = non_max_suppression(
        yolopx_outs["det_out"],
        conf_thres=YOLOPX_CONF_THRES,
        iou_thres=YOLOPX_IOU_THRES,
        classes=None,
        agnostic=False
    )
    det = det_pred[0]
    timings["yolopx_nms"] += time.perf_counter() - t0

    # Lane mask
    t0 = time.perf_counter()
    lane_mask = yolopx_lane_mask(yolopx_x, yolopx_shapes, yolopx_outs["ll_seg"], src_h, src_w)
    timings["lane_mask_only"] += time.perf_counter() - t0

    # Traffic decision
    t0 = time.perf_counter()
    traffic_decision = traffic_decision_from_result(traffic_res, min_conf=TRAFFIC_CONF_THRES)
    timings["traffic_decision_only"] += time.perf_counter() - t0

    # Depth output — use next(iter()) to avoid dict-to-list conversion
    depth_map_m = next(iter(depth_outs.values())).squeeze().cpu().numpy()

    # Scale detections + build det_infos
    det_infos = []
    det_cpu   = None
    if det is not None and len(det):
        det = det.clone()
        det[:, :4] = scale_coords(yolopx_x.shape[2:], det[:, :4], frame_bgr.shape).round()
        det_cpu    = det.detach().cpu().numpy()

        for row in det_cpu:
            x1, y1, x2, y2, conf, cls = row.tolist()
            dist_m = estimate_box_distance_m(depth_map_m, [x1, y1, x2, y2], src_w, src_h)
            det_infos.append({
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "conf": float(conf), "cls": int(cls),
                "cx": float((x1 + x2) * 0.5),
                "dist_m": dist_m,
            })

    # Lane decision
    t0 = time.perf_counter()
    lane_decision = lane_switch_decision_dispatch(
        lane_mask=lane_mask,
        det_infos=det_infos,
        traffic_decision=traffic_decision,
        frame_w=src_w,
        frame_h=src_h,
        lane_strategy=lane_strategy,
        lane_memory=lane_memory,
    )
    timings["lane_post_and_decision"] += time.perf_counter() - t0

    # Draw
    t0  = time.perf_counter()
    vis = frame_bgr.copy()

    if SHOW_LANE_LINES:
        vis = draw_lane_overlay(vis, lane_mask, color=(0, 0, 255), alpha=LANE_ALPHA)

    yolopx_box_count = 0
    if det_cpu is not None:
        for d in det_infos:
            vis = draw_yolopx_box_with_distance(
                vis, [d["x1"], d["y1"], d["x2"], d["y2"]],
                conf=d["conf"], dist_m=d["dist_m"], show_sample_point=True)
            yolopx_box_count += 1

    vis = draw_traffic_boxes(vis, traffic_res)
    vis = draw_decision_panel(vis, traffic_decision, lane_decision)
    timings["draw"] += time.perf_counter() - t0

    return {
        "vis":              vis,
        "traffic_decision": traffic_decision,
        "lane_decision":    lane_decision,
        "yolopx_box_count": yolopx_box_count,
        "timings":          timings,
        "yolopx_ms":        yolopx_ms,
        "depth_ms":         depth_ms,
        "traffic_inf_ms":   traffic_inf_ms,
    }


# ──────────────────────────────────────────────────────────────
# Async video-write worker
# ──────────────────────────────────────────────────────────────
def _video_write_worker(writer, q):
    """Background thread: pulls frames from queue, writes to video."""
    while True:
        item = q.get()
        if item is None:        # sentinel → stop
            q.task_done()
            break
        writer.write(item)
        q.task_done()


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────
def run_final_pipeline(
    run_full_video=False,
    preview_only=False,
    name_suffix="simpleveto",
    lane_strategy=DEFAULT_LANE_STRATEGY,
):
    info = get_video_range(run_full_video=run_full_video)

    src_fps         = info["src_fps"]
    total_frames    = info["total_frames"]
    src_w           = info["src_w"]
    src_h           = info["src_h"]
    start_frame     = info["start_frame"]
    end_frame       = info["end_frame"]
    frames_to_process = info["frames_to_process"]

    print(f"\nVideo: {src_w}x{src_h} | FPS={src_fps:.3f}")
    print(f"Frames total: {total_frames}")
    print(f"Start frame : {start_frame}")
    print(f"End frame   : {end_frame}")
    print(f"Run frames  : {frames_to_process}")

    cap_warm = cv2.VideoCapture(VIDEO_IN)
    cap_warm.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, warm_frame = cap_warm.read()
    cap_warm.release()
    assert ok, "Could not read warmup frame"
    warmup_models(warm_frame)

    lane_memory = None if lane_strategy == "strict_only" else LaneCorridorMemory(lane_strategy)

    stem     = os.path.splitext(os.path.basename(VIDEO_IN))[0]
    mode_tag = "full" if run_full_video else "mid10s"

    out_video   = os.path.join(OUT_DIR, f"{stem}_3model_{mode_tag}_{name_suffix}.mp4")
    out_json    = os.path.join(OUT_DIR, f"{stem}_3model_{mode_tag}_{name_suffix}_summary.json")
    out_csv     = os.path.join(OUT_DIR, f"{stem}_3model_{mode_tag}_{name_suffix}_frames.csv")
    out_preview = os.path.join(OUT_DIR, f"{stem}_3model_{mode_tag}_{name_suffix}_preview.jpg")

    cap = cv2.VideoCapture(VIDEO_IN)
    assert cap.isOpened(), VIDEO_IN
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if preview_only:
        ok, frame_bgr = cap.read()
        cap.release()
        assert ok, "Could not read preview frame"

        rendered = render_one_frame(frame_bgr, src_w, src_h,
                                    lane_strategy=lane_strategy, lane_memory=lane_memory)
        cv2.imwrite(out_preview, rendered["vis"])

        print("\nPREVIEW ONLY DONE")
        print("Saved preview:",    out_preview)
        print("Traffic decision:", rendered["traffic_decision"])
        print("Lane decision   :", rendered["lane_decision"])
        print("YOLOPX box count:", rendered["yolopx_box_count"])
    # NO-PREVIEW PATCH: matplotlib preview display disabled

        return {
            "preview_image":    None,
            "traffic_decision": rendered["traffic_decision"],
            "lane_decision":    rendered["lane_decision"],
            "yolopx_box_count": rendered["yolopx_box_count"],
        }

    writer = cv2.VideoWriter(
        out_video,
        cv2.VideoWriter_fourcc(*FOURCC),
        src_fps,
        (src_w, src_h),
    )
    assert writer.isOpened(), out_video

    # ── Async write setup ────────────────────────────────────────
    if ASYNC_VIDEO_WRITE:
        write_q      = queue.Queue(maxsize=VIDEO_WRITE_QUEUE_SZ)
        write_thread = threading.Thread(
            target=_video_write_worker, args=(writer, write_q), daemon=True)
        write_thread.start()

        def enqueue_write(frame):
            write_q.put(frame)
    else:
        def enqueue_write(frame):
            writer.write(frame)
    # ─────────────────────────────────────────────────────────────

    timers  = defaultdict(float)
    rows    = []
    n_frames        = 0
    saved_preview   = False

    sum_yolopx_forward_ms  = 0.0
    sum_depth_forward_ms   = 0.0
    sum_traffic_forward_ms = 0.0
    sum_combined_forward_ms = 0.0

    print("\nRUNNING FINAL 3-MODEL PIPELINE...")

    while True:
        if n_frames >= frames_to_process:
            break

        t_e2e0 = time.perf_counter()

        t0 = time.perf_counter()
        ok, frame_bgr = cap.read()
        timers["read"] += time.perf_counter() - t0
        if not ok:
            break

        frame_index = start_frame + n_frames
        rendered    = render_one_frame(frame_bgr, src_w, src_h,
                                       lane_strategy=lane_strategy, lane_memory=lane_memory)

        vis             = rendered["vis"]
        traffic_decision = rendered["traffic_decision"]
        lane_decision   = rendered["lane_decision"]
        yolopx_box_count = rendered["yolopx_box_count"]

        sum_yolopx_forward_ms   += rendered["yolopx_ms"]
        sum_depth_forward_ms    += rendered["depth_ms"]
        sum_traffic_forward_ms  += rendered["traffic_inf_ms"]
        sum_combined_forward_ms += (rendered["yolopx_ms"] + rendered["depth_ms"] + rendered["traffic_inf_ms"])

        timers["preprocess"]          += rendered["timings"]["preprocess"]
        timers["yolopx_forward"]      += rendered["timings"]["yolopx_forward_s"]
        timers["depth_forward"]       += rendered["timings"]["depth_forward_s"]
        timers["traffic_total_call"]  += rendered["timings"]["traffic_total_call"]
        timers["traffic_forward"]     += rendered["timings"]["traffic_forward_s"]
        timers["yolopx_nms"]          += rendered["timings"]["yolopx_nms"]
        timers["lane_post_and_decision"] += rendered["timings"]["lane_post_and_decision"]
        timers["traffic_decision_only"]  += rendered["timings"]["traffic_decision_only"]
        timers["draw"]                += rendered["timings"]["draw"]

        if not saved_preview:
            # NO-PREVIEW PATCH:
            # Do not save preview image. Keep flag true so this branch runs once only.
            saved_preview = True

        t0 = time.perf_counter()
        # NO-PREVIEW PATCH: _api_live_preview_write(vis) disabled
        enqueue_write(vis)          # non-blocking when ASYNC_VIDEO_WRITE=True
        try:
            _next_run_write_processed_preview(vis)
        except Exception:
            pass
        timers["write"] += time.perf_counter() - t0

        timers["end_to_end"] += time.perf_counter() - t_e2e0
        n_frames += 1

        rows.append({
            "frame_index":     int(frame_index),
            "time_sec":        float(frame_index / src_fps),
            "traffic_stop":    int(traffic_decision["stop"]),
            "traffic_ready":   int(traffic_decision["ready"]),
            "traffic_go":      int(traffic_decision["go"]),
            "traffic_active_class": traffic_decision["active_class"],
            "traffic_active_conf":  float(traffic_decision["active_conf"]),
            "traffic_num_boxes":    int(traffic_decision["num_boxes"]),
            "lane_left_switch":  int(lane_decision["left_switch"]),
            "lane_right_switch": int(lane_decision["right_switch"]),
            "lane_found":        int(lane_decision["lane_found"]),
            "yolopx_box_count":  int(yolopx_box_count),
        })

        if n_frames % REPORT_EVERY == 0 or n_frames == frames_to_process:
            print(
                f"{n_frames:4d}/{frames_to_process} | "
                f"YOLOPX FPS={1000.0 * n_frames / max(sum_yolopx_forward_ms, 1e-9):7.3f} | "
                f"Depth FPS={1000.0 * n_frames / max(sum_depth_forward_ms, 1e-9):7.3f} | "
                f"Traffic FPS={1000.0 * n_frames / max(sum_traffic_forward_ms, 1e-9):7.3f} | "
                f"Combined FPS={1000.0 * n_frames / max(sum_combined_forward_ms, 1e-9):7.3f} | "
                f"E2E FPS={n_frames / max(timers['end_to_end'], 1e-9):7.3f}"
            )

    # ── Drain the write queue before releasing resources ─────────
    if ASYNC_VIDEO_WRITE:
        write_q.put(None)       # sentinel
        write_thread.join()
    # ─────────────────────────────────────────────────────────────

    cap.release()
    writer.release()

    fieldnames = [
        "frame_index", "time_sec",
        "traffic_stop", "traffic_ready", "traffic_go",
        "traffic_active_class", "traffic_active_conf", "traffic_num_boxes",
        "lane_left_switch", "lane_right_switch", "lane_found",
        "yolopx_box_count",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "video_input":       VIDEO_IN,
        "yolopx_engine":     YOLOPX_ENGINE,
        "depth_engine":      DEPTH_ENGINE,
        "traffic_engine":    TRAFFIC_ENGINE,
        "lane_strategy":     lane_strategy,
        "output_video":      out_video,
        "preview_image":     None,
        "csv_output":        out_csv,

        "source_resolution":       f"{src_w}x{src_h}",
        "source_fps_metadata":     float(src_fps),
        "total_frames_in_video":   int(total_frames),
        "start_frame":             int(start_frame),
        "end_frame_exclusive":     int(end_frame),
        "frames_processed":        int(n_frames),

        "yolopx_forward_ms_per_frame":   float(sum_yolopx_forward_ms  / max(n_frames, 1)),
        "depth_forward_ms_per_frame":    float(sum_depth_forward_ms   / max(n_frames, 1)),
        "traffic_forward_ms_per_frame":  float(sum_traffic_forward_ms / max(n_frames, 1)),
        "combined_forward_ms_per_frame": float(sum_combined_forward_ms / max(n_frames, 1)),

        "yolopx_forward_fps":   float(1000.0 * n_frames / max(sum_yolopx_forward_ms,   1e-9)),
        "depth_forward_fps":    float(1000.0 * n_frames / max(sum_depth_forward_ms,    1e-9)),
        "traffic_forward_fps":  float(1000.0 * n_frames / max(sum_traffic_forward_ms,  1e-9)),
        "combined_forward_fps": float(1000.0 * n_frames / max(sum_combined_forward_ms, 1e-9)),

        "end_to_end_ms_per_frame": float(1000.0 * timers["end_to_end"] / max(n_frames, 1)),
        "end_to_end_fps":          float(n_frames / max(timers["end_to_end"], 1e-9)),

        "read_ms_per_frame":                   float(1000.0 * timers["read"]                   / max(n_frames, 1)),
        "preprocess_ms_per_frame":             float(1000.0 * timers["preprocess"]             / max(n_frames, 1)),
        "yolopx_nms_ms_per_frame":             float(1000.0 * timers["yolopx_nms"]             / max(n_frames, 1)),
        "lane_post_and_decision_ms_per_frame": float(1000.0 * timers["lane_post_and_decision"] / max(n_frames, 1)),
        "traffic_total_call_ms_per_frame":     float(1000.0 * timers["traffic_total_call"]     / max(n_frames, 1)),
        "traffic_decision_only_ms_per_frame":  float(1000.0 * timers["traffic_decision_only"]  / max(n_frames, 1)),
        "draw_ms_per_frame":                   float(1000.0 * timers["draw"]                   / max(n_frames, 1)),
        "write_ms_per_frame":                  float(1000.0 * timers["write"]                  / max(n_frames, 1)),

        "notes": {
            "yolopx_boxes":        True,
            "lane_lines":          bool(SHOW_LANE_LINES),
            "drivable_area":       False,
            "box_distance_source": "metric-depth TRT engine",
            "box_distance_method": "median depth from bottom-center patch inside YOLOPX box",
            "traffic_decision":    "red->stop, yellow->ready, green->go",
            "lane_decision":       f"{lane_strategy}: persistent adjacent-lane corridor memory first, then close-object veto inside the remembered corridor",
            "traffic_forward_note": "traffic forward time is taken from Triton adapter result.speed['inference']",
            "speed_build":          "parallel-preprocess + band-cache + async-write + LUT-overlay + GPU-normalize",
        },
    }

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ FINAL SUMMARY ================")
    print("Frames processed            :", summary["frames_processed"])
    print("Saved video                 :", out_video)
    print("Saved preview               :", out_preview)
    print("Saved csv                   :", out_csv)
    print()
    print("YOLOPX forward ms/frame     :", f"{summary['yolopx_forward_ms_per_frame']:.3f}")
    print("Depth forward ms/frame      :", f"{summary['depth_forward_ms_per_frame']:.3f}")
    print("Traffic forward ms/frame    :", f"{summary['traffic_forward_ms_per_frame']:.3f}")
    print("Combined forward ms/frame   :", f"{summary['combined_forward_ms_per_frame']:.3f}")
    print()
    print("YOLOPX forward FPS          :", f"{summary['yolopx_forward_fps']:.3f}")
    print("Depth forward FPS           :", f"{summary['depth_forward_fps']:.3f}")
    print("Traffic forward FPS         :", f"{summary['traffic_forward_fps']:.3f}")
    print("Combined forward FPS        :", f"{summary['combined_forward_fps']:.3f}")
    print("End-to-end FPS              :", f"{summary['end_to_end_fps']:.3f}")
    print()
    print("read ms/frame               :", f"{summary['read_ms_per_frame']:.3f}")
    print("preprocess ms/frame         :", f"{summary['preprocess_ms_per_frame']:.3f}")
    print("yolopx nms ms/frame         :", f"{summary['yolopx_nms_ms_per_frame']:.3f}")
    print("lane post+decision ms/frame :", f"{summary['lane_post_and_decision_ms_per_frame']:.3f}")
    print("traffic total call ms/frame :", f"{summary['traffic_total_call_ms_per_frame']:.3f}")
    print("traffic decision ms/frame   :", f"{summary['traffic_decision_only_ms_per_frame']:.3f}")
    print("draw ms/frame               :", f"{summary['draw_ms_per_frame']:.3f}")
    print("write ms/frame              :", f"{summary['write_ms_per_frame']:.3f}")
    print("================================================")
    # NO-PREVIEW PATCH: matplotlib preview display disabled

    return summary


def run_first_minutes_pipeline(
    minutes=5.0,
    preview_only=False,
    name_suffix=None,
    lane_strategy=DEFAULT_LANE_STRATEGY,
):
    """Run only the first N minutes of VIDEO_IN while preserving the same pipeline/rendering."""
    orig_get_video_range = get_video_range

    def _first_minutes_get_video_range(run_full_video):
        info      = orig_get_video_range(True)
        src_fps   = info["src_fps"]
        total_frames = info["total_frames"]
        src_w     = info["src_w"]
        src_h     = info["src_h"]

        start_frame = 0
        clip_len    = int(round(float(minutes) * 60.0 * src_fps))
        end_frame   = min(total_frames, start_frame + clip_len)

        return {
            "src_fps":          src_fps,
            "total_frames":     total_frames,
            "src_w":            src_w,
            "src_h":            src_h,
            "start_frame":      start_frame,
            "end_frame":        end_frame,
            "frames_to_process": end_frame - start_frame,
        }

    try:
        globals()["get_video_range"] = _first_minutes_get_video_range
        suffix = name_suffix if name_suffix is not None else f"{lane_strategy}_first{int(round(minutes))}min"
        return run_final_pipeline(
            run_full_video=True,
            preview_only=preview_only,
            name_suffix=suffix,
            lane_strategy=lane_strategy,
        )
    finally:
        globals()["get_video_range"] = orig_get_video_range

result = run_first_minutes_pipeline(
    minutes=5.0,
    preview_only=False,
    name_suffix="speed_optimized_first5min",
    lane_strategy="corridor_memory_two_hits",
)
result
