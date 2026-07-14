import numpy as np

# ORIGINAL (verbatim from your locked pipeline)
def _find_peaks_strict_ORIG(hist, thr, min_sep):
    peaks = []
    n = len(hist)
    for i in range(1, n - 1):
        if hist[i] >= thr and hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]:
            peaks.append(i)
    peaks = sorted(peaks, key=lambda i: hist[i], reverse=True)
    keep = []
    for p in peaks:
        if all(abs(p - q) >= min_sep for q in keep):
            keep.append(p)
    return sorted(keep)

# VECTORIZED (proposed replacement)
def _find_peaks_strict_VEC(hist, thr, min_sep):
    n = len(hist)
    if n < 3:
        return []
    h = np.asarray(hist)
    center, left, right = h[1:-1], h[:-2], h[2:]
    is_peak = (center >= thr) & (center >= left) & (center >= right)
    idx = np.nonzero(is_peak)[0] + 1
    if idx.size == 0:
        return []
    order = np.argsort(-h[idx], kind="stable")
    candidates = idx[order].tolist()
    keep = []
    for p in candidates:
        if all(abs(p - q) >= min_sep for q in keep):
            keep.append(p)
    return sorted(keep)

rng = np.random.default_rng(42)
n_tests = 20000
mismatches = 0
tested_shapes = 0

for trial in range(n_tests):
    n = rng.integers(3, 1930)          # cover widths up to 1920 (your video width) + edge sizes
    kind = trial % 6
    if kind == 0:
        hist = np.zeros(n, dtype=np.float32)                          # all-zero
    elif kind == 1:
        hist = np.full(n, rng.uniform(0, 5), dtype=np.float32)        # flat plateau (ties)
    elif kind == 2:
        hist = rng.integers(0, 4, size=n).astype(np.float32)          # low-cardinality (many ties)
    elif kind == 3:
        hist = rng.random(n).astype(np.float32) * 50                  # generic random
    elif kind == 4:
        # smoothed-like signal, similar shape to real np.convolve output
        raw = rng.random(n).astype(np.float32)
        k = np.ones(11) / 11
        hist = np.convolve(raw, k, mode="same").astype(np.float32)
    else:
        hist = np.concatenate([np.zeros(n // 2), rng.random(n - n // 2) * 10]).astype(np.float32)

    thr = rng.uniform(0, hist.max() if hist.max() > 0 else 1.0)
    min_sep = int(rng.integers(1, max(2, n // 8)))

    a = _find_peaks_strict_ORIG(hist, thr, min_sep)
    b = _find_peaks_strict_VEC(hist, thr, min_sep)

    tested_shapes += 1
    if a != b:
        mismatches += 1
        if mismatches <= 5:
            print("MISMATCH", "n=", n, "thr=", thr, "min_sep=", min_sep)
            print(" orig:", a)
            print(" vec :", b)

print(f"\nTotal cases tested: {tested_shapes}")
print(f"Mismatches: {mismatches}")
print("RESULT:", "PASS - bit-identical on all cases" if mismatches == 0 else "FAIL - see mismatches above")
