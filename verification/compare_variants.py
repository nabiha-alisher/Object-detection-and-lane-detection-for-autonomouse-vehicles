"""
Run this on RunPod after producing frames CSVs / summary JSONs from the matrix runner.

Usage:
    cd /workspace/fyp
    python3 compare_variants.py --output-dir output/FINAL_3MODEL_UNIFIED --baseline http_seq

What it does:
  1. Finds every *_frames.csv in --output-dir.
  2. Picks the file matching --baseline as the reference for decision correctness.
  3. For every other CSV, checks lane_left_switch / lane_right_switch / lane_found /
     traffic_stop / traffic_ready / traffic_go / yolopx_box_count are IDENTICAL,
     row for row, against the baseline. Any mismatch is printed with frame_index.
  4. Prints a combined FPS/timing table from every matching *_summary.json.

This is the same check I ran on your first handoff (0 mismatches across all 4
original variants). Re-run it every time you add a new variant so nothing
silently drifts.
"""
import argparse
import glob
import json
import os
import sys

try:
    import pandas as pd
except ImportError:
    print("pandas required: pip install pandas --break-system-packages")
    sys.exit(1)

DECISION_COLS = [
    "lane_left_switch", "lane_right_switch", "lane_found",
    "traffic_stop", "traffic_ready", "traffic_go",
    "traffic_active_class", "yolopx_box_count",
]

TIMING_KEYS = [
    "end_to_end_fps", "end_to_end_ms_per_frame",
    "combined_forward_ms_per_frame",
    "yolopx_forward_ms_per_frame", "depth_forward_ms_per_frame",
    "traffic_forward_ms_per_frame", "traffic_total_call_ms_per_frame",
    "lane_post_and_decision_ms_per_frame", "draw_ms_per_frame",
    "write_ms_per_frame",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--baseline", default="http_seq",
                     help="substring identifying the baseline CSV filename, default 'http_seq'")
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.output_dir, "*_frames.csv")))
    if not csvs:
        print("No *_frames.csv found in", args.output_dir)
        sys.exit(1)

    baseline_path = next((c for c in csvs if args.baseline in os.path.basename(c)), None)
    if baseline_path is None:
        print(f"No CSV matching baseline tag '{args.baseline}' found among:")
        for c in csvs:
            print(" ", c)
        sys.exit(1)

    print("Baseline:", baseline_path)
    base_df = pd.read_csv(baseline_path)

    print("\n" + "=" * 90)
    print("DECISION EQUALITY CHECK (vs baseline)")
    print("=" * 90)

    any_fail = False
    for c in csvs:
        if c == baseline_path:
            continue
        df = pd.read_csv(c)
        n = min(len(base_df), len(df))
        if len(base_df) != len(df):
            print(f"[WARN] {os.path.basename(c)}: row count differs "
                  f"(baseline={len(base_df)}, this={len(df)}) — comparing first {n} rows")

        cols = [col for col in DECISION_COLS if col in base_df.columns and col in df.columns]
        b = base_df[cols].iloc[:n].reset_index(drop=True)
        d = df[cols].iloc[:n].reset_index(drop=True)
        diff_mask = (b != d).any(axis=1)
        mism = int(diff_mask.sum())

        status = "PASS" if mism == 0 else "FAIL"
        if mism != 0:
            any_fail = True
        print(f"{status:4s}  {os.path.basename(c):55s} mismatched_rows={mism}/{n}")

        if mism != 0:
            bad_idx = base_df.loc[diff_mask.index[diff_mask]].index[:5]
            for i in bad_idx:
                print(f"       frame_index={base_df.loc[i, 'frame_index'] if 'frame_index' in base_df else i}")
                print("        baseline:", b.iloc[i].to_dict())
                print("        this    :", d.iloc[i].to_dict())

    print("\n" + "=" * 90)
    print("FPS / TIMING TABLE (from *_summary.json in same dir)")
    print("=" * 90)

    jsons = sorted(glob.glob(os.path.join(args.output_dir, "*_summary.json")))
    header = f"{'file':45s}" + "".join(f"{k[:14]:>16s}" for k in TIMING_KEYS)
    print(header)
    for j in jsons:
        data = json.load(open(j))
        row = f"{os.path.basename(j)[:45]:45s}"
        for k in TIMING_KEYS:
            v = data.get(k)
            row += f"{v:16.3f}" if isinstance(v, (int, float)) else f"{'NA':>16s}"
        print(row)

    print("\nOVERALL:", "SOME MISMATCHES FOUND - DO NOT TRUST THESE VARIANTS" if any_fail else "ALL VARIANTS DECISION-IDENTICAL TO BASELINE")


if __name__ == "__main__":
    main()
