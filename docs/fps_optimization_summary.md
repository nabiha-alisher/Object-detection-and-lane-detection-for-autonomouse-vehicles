# FPS Optimization Summary

This document summarizes the FPS optimization work performed on the three-model AI traffic assistance pipeline.

## 1. Baseline and Optimization Goal

The original system used a Triton HTTP-based sequential inference pipeline. The goal was to improve end-to-end FPS while preserving the same decision-level assistance behavior.

The three models used in the pipeline are:

```text
YOLOPX                 Lane-line segmentation and object/vehicle detection
DepthAnythingV2         Metric depth estimation
Traffic-light model     Traffic signal detection
```

The same 5-minute video section was used for profiler comparisons:

```text
9000 frames
same input video
same lane strategy
same output structure
```

## 2. Original Four Profiler Variants

The original four profiler files were:

```text
fps_http_seq.py
fps_grpc_seq.py
fps_http_parallel.py
fps_grpc_parallel.py
```

Meaning:

```text
fps_http_seq.py       Triton HTTP + sequential model calls
fps_grpc_seq.py       Triton gRPC + sequential model calls
fps_http_parallel.py  Triton HTTP + parallel model calls
fps_grpc_parallel.py  Triton gRPC + parallel model calls
```

## 3. Original Four-Profiler Results

| Variant | Description | FPS | E2E ms/frame | Model ms/frame | Lane ms/frame | Write ms/frame |
|---|---|---:|---:|---:|---:|---:|
| HTTP Sequential | Original Triton HTTP baseline | 6.648 | 150.418 | 66.859 | 45.565 | 0.144 |
| gRPC Sequential | HTTP replaced with gRPC | 9.317 | 107.332 | 32.174 | 39.649 | 0.119 |
| HTTP Parallel | HTTP with parallel model calls | 6.630 | 150.830 | 129.325 | 63.786 | 0.145 |
| gRPC Parallel | gRPC with parallel model calls | 9.764 | 102.422 | 45.424 | 49.001 | 0.120 |

Key observations:

```text
HTTP sequential to gRPC sequential: 6.648 FPS to 9.317 FPS
This was about a 40.1% FPS improvement.
```

```text
gRPC sequential to gRPC parallel: 9.317 FPS to 9.764 FPS
This was a smaller improvement of about 4.8%.
```

```text
HTTP parallel did not improve the pipeline.
```

After these tests, lane post-processing was identified as a major bottleneck.

## 4. gRPC Parallel + Vectorized Lane Optimization

The optimized Triton profiler file was:

```text
fps_grpc_parallel_v2.py
```

This version represents:

```text
Triton gRPC + parallel inference + vectorized lane logic
```

The lane peak-detection step was optimized using NumPy vectorization. The older implementation used repeated Python-level checks over histogram positions. The optimized version detects candidate peaks using vectorized neighbor comparisons and then keeps the existing greedy minimum-separation behavior to preserve decision logic.

Result:

| Variant | FPS | E2E ms/frame | Model ms/frame | Lane ms/frame | Draw ms/frame | Write ms/frame |
|---|---:|---:|---:|---:|---:|---:|
| Old gRPC Parallel | 9.764 | 102.422 | 45.424 | 49.001 | 10.180 | 0.120 |
| gRPC Parallel + Vectorized Lane | 15.009 | 66.628 | 44.600 | 3.178 | 19.288 | 0.151 |

Improvement:

```text
FPS: 9.764 to 15.009
Approximate gain: +53.7%
```

```text
Lane post-processing: 49.001 ms/frame to 3.178 ms/frame
Approximate lane-processing speedup: 15.4x
```

The model time remained almost the same:

```text
45.424 ms/frame to 44.600 ms/frame
```

This confirms that the major gain came from optimizing lane post-processing, not from random model-serving variation.

## 5. Decision-Level Verification

The old gRPC parallel CSV and the new gRPC parallel vectorized CSV were compared over 9000 frames on selected decision columns:

```text
frame_index
traffic_stop
traffic_ready
traffic_go
traffic_active_class
traffic_num_boxes
lane_left_switch
lane_right_switch
lane_found
yolopx_box_count
```

Result:

```text
Old rows: 9000
New rows: 9000
Compared rows: 9000
Mismatched decision rows: 0 / 9000
Result: Decision identical
```

This verifies decision-level equivalence for the selected CSV decision fields.

## 6. Raw TensorRT + Vectorized Lane Result

A Raw TensorRT version with the same vectorized lane logic was also tested:

```text
fps_rawtrt_seq_v2.py
```

Result:

| Variant | FPS | E2E ms/frame | Model ms/frame | Lane ms/frame | Draw ms/frame | Write ms/frame |
|---|---:|---:|---:|---:|---:|---:|
| Raw TensorRT + Vectorized Lane | 24.102 | approximately 41.49 | 5.522 | 2.087 | 11.395 | 0.036 |

This was the fastest local inference result. It demonstrates the maximum speed potential of the project when avoiding Triton serving overhead.

## 7. Final FPS Progression

| Stage | FPS |
|---|---:|
| HTTP Sequential baseline | 6.648 |
| gRPC Sequential | 9.317 |
| gRPC Parallel | 9.764 |
| gRPC Parallel + Vectorized Lane | 15.009 |
| Raw TensorRT + Vectorized Lane | 24.102 |

## 8. Final Conclusion

The final optimized Triton-serving variant achieved 15.009 FPS using gRPC parallel inference and vectorized lane post-processing. This was a 53.7% improvement over the previous gRPC parallel version and approximately 2.26x faster than the original HTTP sequential baseline.

The Raw TensorRT + vectorized lane variant achieved 24.102 FPS and serves as the fastest local inference reference. The Triton version remains important because it demonstrates deployable model serving, while the Raw TensorRT version demonstrates maximum local performance.
