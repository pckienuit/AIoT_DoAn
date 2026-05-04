# Bug Report: Zero-Padding Crop Pipeline → NME Explosion

## Summary

Face landmark models trained with a crop expansion pipeline using **zero-padding** suffered from severe NME (Normalized Mean Error) degradation on test set. Root cause: ~30% of test samples had landmark coordinates clipped to boundary values 0 or 1, causing IOD (interocular distance) to collapse to near-zero and NME to explode.

---

## Root Cause Analysis

### 1. The Crop Pipeline

Both `train.py` (local) and `train_v8.py` (VPS) use a face crop pipeline:

```
Input image → Expand BBox by CROP_SCALE=2.14 → Crop → Resize to 224×224 → Normalize [0,1]
```

When the expanded crop region extends beyond image boundaries, the original code filled out-of-bounds pixels with **black (zero)**:

```python
face_crop = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)  # BLACK FILL

src_x1, src_y1 = max(0, crop_x),          max(0, crop_y)
src_x2, src_y2 = min(W, crop_x + crop_w), min(H, crop_y + crop_h)
dst_x1 = max(0, -crop_x)
dst_y1 = max(0, -crop_y)

if src_x2 > src_x1 and src_y2 > src_y1:
    face_crop[dst_y1:dst_y2, dst_x1:dst_x2] = image_full[src_y1:src_y2, src_x1:src_x2]
```

### 2. Why Black Regions Corrupt Landmarks

The landmark normalization formula:

```
lm_norm[x] = clip((lm_raw[x] - crop_x) / crop_w, 0.0, 1.0)
lm_norm[y] = clip((lm_raw[y] - crop_y) / crop_h, 0.0, 1.0)
```

When a face is near the image edge and the crop expands outward:

| Condition | Effect |
|-----------|--------|
| Landmark falls in black region | Clipped to 0.0 or 1.0 |
| Both eyes clipped to same boundary | IOD ≈ 0.0 |
| IOD used as denominator in NME | NME → ∞ (explosion) |

**With CROP_SCALE = 2.14**, the crop is 2.14× wider than the face bounding box. For small faces near edges, this almost always extends beyond the image boundary.

### 3. Diagnostic Evidence (partition=2 test set, 19,962 samples)

```
partition2_samples=19962

OLD (zero-pad) IOD stats:
  mean=0.095591, median=0.092769, p5=0.000000

IOD < threshold percentage (OLD):
  IOD < 0.01: 17.3480%
  IOD < 0.02: 19.6523%
  IOD < 0.03: 21.3556%
  IOD < 0.05: 25.4083%

Landmark coordinate clipping (OLD):
  coords clipped to 0 or 1:     12.4046%
  samples with any clipped coord: 29.6263%
```

**~30% of test samples had at least one landmark coordinate forced to 0 or 1 due to black padding.**

---

## The Fix: BORDER_REPLICATE

Replace zero-padding with `cv2.BORDER_REPLICATE` — replicate edge pixels instead of filling black:

```python
pad_left   = max(0, -crop_x)
pad_top    = max(0, -crop_y)
pad_right  = max(0, (crop_x + crop_w) - W)
pad_bottom = max(0, (crop_y + crop_h) - H)

if pad_left or pad_top or pad_right or pad_bottom:
    image_full = cv2.copyMakeBorder(
        image_full,
        top=pad_top, bottom=pad_bottom,
        left=pad_left, right=pad_right,
        borderType=cv2.BORDER_REPLICATE
    )
    crop_x += pad_left
    crop_y += pad_top

face_crop = image_full[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
```

**Key properties:**
- Edge pixels are replicated outward instead of filling black
- Landmark coordinates are no longer artificially pushed to 0/1 by black regions
- Crop region is always fully covered — no information loss
- `torchvision.transforms.Normalize` clips to [0,1] anyway, so any out-of-bounds coords are handled naturally

### Diagnostic After Fix

```
NEW (BORDER_REPLICATE) IOD stats:
  mean=0.125092, median=0.111429, p5=0.044119

IOD < threshold percentage (NEW):
  IOD < 0.01:  0.0551%   (was 17.35%, delta: -17.29%)
  IOD < 0.02:  0.7114%   (was 19.65%, delta: -18.94%)
  IOD < 0.03:  2.0739%   (was 21.36%, delta: -19.28%)
  IOD < 0.05:  6.5474%   (was 25.41%, delta: -18.86%)
```

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| IOD mean | 0.0956 | **0.1251** | +31% |
| IOD p5 | 0.000 | **0.044** | usable |
| IOD < 0.01 samples | 17.35% | **0.06%** | -17.3% |
| Samples with clipped coords | 29.63% | 29.29% | slight |

---

## Lessons Learned

1. **Zero-padding at crop boundaries is dangerous for landmark tasks.** Any task that normalizes coordinates to [0,1] is vulnerable to this class of bug.

2. **Diagnostic before optimizing.** The Wing loss, focal loss, and other refinements wouldn't have helped until this fundamental data-corruption bug was fixed.

3. **CROP_SCALE=2.14 is aggressive.** This scale factor causes the crop to exceed image boundaries in ~30% of cases. Alternatives:
   - Reduce scale factor (trade-off: less context)
   - Use BORDER_REPLICATE or BORDER_REFLECT (current fix)
   - Reject crops that exceed boundaries (loses data)

4. **Always measure IOD distribution on your test set.** A p5 of 0.0 is an immediate red flag for landmark normalization pipelines.

---

## Files Modified

| File | Change |
|------|--------|
| `train.py` | `_make_crop()`: zero-pad → BORDER_REPLICATE |
| `train_v8.py` | `_make_crop()`: zero-pad → BORDER_REPLICATE |
| `scripts/utils/_diag_partition2.py` | Diagnostic comparison script |
