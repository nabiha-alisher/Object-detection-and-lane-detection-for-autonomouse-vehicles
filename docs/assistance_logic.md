# Assistance Logic / ADAS Decision Flow

This document explains the final ADAS assistance-triggering logic used in the FYP traffic assistance system.

## 1. Meaning of ADAS Logic in This Project

In this project, ADAS logic means the rule-based decision layer that converts perception model outputs into final driving-assistance flags.

The models produce perception outputs, but the ADAS logic decides whether assistance should be triggered.

Final assistance flags:

```text
traffic_stop
traffic_ready
traffic_go
lane_left_switch
lane_right_switch
lane_found
```

The final assistance logic is based on fusion of:

```text
YOLOPX lane-line segmentation
YOLOPX object detections
DepthAnythingV2 metric depth estimation
Traffic-light model output
Rule-based lane, traffic, memory, and object-blocking logic
```

## 2. Overall Flow

```text
Input video frame
        ↓
Preprocess frame for all three models
        ↓
Run three models
        ├── YOLOPX
        ├── DepthAnythingV2
        └── Traffic-light model
        ↓
Post-process model outputs
        ↓
Traffic-light decision
        ↓
Lane/corridor decision
        ↓
Depth-based object blocker check
        ↓
Final assistance flags
        ↓
Video overlay + CSV logging + JSON summary
```

In the optimized Triton version, the model calls are executed using gRPC parallel inference, but the decision meaning remains the same.

## 3. Traffic-Light Assistance Logic

The traffic-light model detects four classes:

```text
tl_red
tl_yellow
tl_green
tl_none
```

The decision logic ignores:

```text
tl_none
low-confidence detections below 0.25
```

It selects the highest-confidence valid traffic-light class.

Decision rules:

```text
If best class = tl_red:
    traffic_stop = 1
    traffic_ready = 0
    traffic_go = 0
```

```text
If best class = tl_yellow:
    traffic_stop = 0
    traffic_ready = 1
    traffic_go = 0
```

```text
If best class = tl_green:
    traffic_stop = 0
    traffic_ready = 0
    traffic_go = 1
```

```text
If no valid traffic light is detected:
    traffic_stop = 0
    traffic_ready = 0
    traffic_go = 0
```

Meaning:

```text
Red light    → stop assistance
Yellow light → ready/caution assistance
Green light  → go assistance
No light     → no traffic-light assistance
```

## 4. Lane Assistance Logic

YOLOPX outputs a lane-line segmentation mask:

```text
ll_seg → lane_mask
```

The system analyzes the lower road region of the image:

```text
58% to 95% of frame height
```

This region is split into:

```text
10 horizontal bands
```

For each band, the system builds a horizontal lane-pixel histogram and detects lane peaks.

## 5. Vectorized Lane Peak Detection

The final optimized version vectorizes the lane peak detection stage.

Simplified logic:

```text
Convert histogram to NumPy array
Compare each center value with its left and right neighbors
Find valid candidate peaks at once
Sort candidates by histogram strength
Apply greedy minimum-separation filtering
Return stable peak positions
```

This reduces the computational cost of lane post-processing while preserving the same decision-level behavior.

## 6. Ego Lane Detection

The system estimates the current ego lane using lane peaks around the image center.

```text
left_near  = nearest lane peak to the left of center
right_near = nearest lane peak to the right of center
```

The ego lane is valid only if there is enough band support.

Minimum ego support:

```text
5 bands
```

The system calculates:

```text
left_current  = median left boundary
right_current = median right boundary
ego_width     = right_current - left_current
```

The ego lane width must be reasonable:

```text
minimum width = 18% of frame width
maximum width = 55% of frame width
```

If valid:

```text
lane_found = 1
```

If invalid:

```text
lane_found = 0
lane_left_switch = 0
lane_right_switch = 0
```

## 7. Adjacent Lane / Corridor Detection

After the ego lane is found, the system searches for adjacent lane corridors.

```text
left_far  = lane peak further left of the ego lane
right_far = lane peak further right of the ego lane
```

The adjacent lane gap must be realistic:

```text
0.45 × ego lane width ≤ adjacent gap ≤ 1.80 × ego lane width
```

Minimum adjacent support:

```text
4 bands
```

This produces possible left and right lane availability, but these are not final until memory, geometry, and blocker checks are applied.

## 8. Corridor Memory Strategy

The final lane strategy is:

```text
corridor_memory_two_hits
```

It stabilizes lane decisions across frames so that the system does not flicker when lane markings briefly disappear.

Important settings:

```text
max_miss_frames = 5
forget_after_miss = 7
recent_window = 6
min_recent_hits = 2
```

Meaning:

```text
If a lane corridor briefly disappears, the system can remember it.
If it disappears for too long, the memory is cleared.
```

## 9. Fresh Lower Attachment Check

Even if memory says a lane exists, the system checks whether the side lane is freshly attached in the lower part of the frame.

Lower region starts around:

```text
74% of frame height
```

This check prevents stale memory or upper-frame noise from causing false lane-switch availability.

If fresh lower attachment is not confirmed:

```text
lane_left_switch or lane_right_switch = 0
```

## 10. Boundary Impossible Check

The system checks whether an adjacent corridor is geometrically impossible by testing:

```text
reasonable width
reasonable inner gap from ego lane
reasonable center offset
```

If the side corridor is geometrically impossible:

```text
lane_left_switch or lane_right_switch = 0
```

## 11. Depth-Based Object Blocker Logic

YOLOPX detects objects and vehicles. DepthAnythingV2 estimates distance.

For each detected object, the system samples the lower-middle region of the bounding box:

```text
x range: 30% to 70% of box width
y range: 55% to 90% of box height
```

It then calculates the median valid depth.

Valid depth range:

```text
0.1 m to 80.0 m
```

An object is considered close if:

```text
distance ≤ 13.5 m
```

If a close object lies inside the target left or right lane corridor, the lane-switch suggestion is blocked.

```text
close object in target corridor → lane_left_switch/right_switch = 0
```

## 12. Final Lane Switch Decision

Final left or right lane assistance is triggered only if all conditions pass:

```text
If ego lane is not found:
    switch = 0
Else if adjacent lane does not exist:
    switch = 0
Else if adjacent corridor is geometrically impossible:
    switch = 0
Else if fresh lower attachment is not confirmed:
    switch = 0
Else if close object blocks the side corridor:
    switch = 0
Else:
    switch = 1
```

Meaning:

```text
lane_left_switch = 1
```

means the left lane/corridor is available, geometrically valid, stable/fresh, and not blocked by a close object.

```text
lane_right_switch = 1
```

means the right lane/corridor is available, geometrically valid, stable/fresh, and not blocked by a close object.

## 13. Final Thesis Wording

The final ADAS logic is based on rule-based fusion of three model outputs. YOLOPX provides object detections and lane-line segmentation, DepthAnythingV2 provides metric distance estimation, and the traffic-light model provides the active traffic-light class. The traffic assistance flags are triggered by selecting the highest-confidence valid traffic-light class: red activates the stop flag, yellow activates the ready/caution flag, and green activates the go flag.

For lane assistance, the YOLOPX lane-line mask is analyzed in the lower road region of the frame. The region is divided into horizontal bands, lane-pixel histograms are generated, and lane peaks are detected using vectorized NumPy-based peak detection. These peaks are used to estimate the ego lane and possible adjacent left/right lane corridors. The corridor-memory strategy stabilizes the decision across frames, while fresh lower attachment and geometry checks prevent false lane availability. Finally, YOLOPX object detections are combined with DepthAnythingV2 distance estimates to block a lane-switch suggestion if a close object is present in the target corridor. Therefore, the final ADAS output is generated through the combined interpretation of traffic-light state, lane geometry, object detection, and depth-based distance information.
