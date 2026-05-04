import sys, os
import numpy as np
import pandas as pd

# Auto-detect dataset paths
def _find_csv(path):
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            if f == "labels.csv":
                return os.path.join(dirpath, f)
    return None

def _find_img(path):
    for dirpath, dirnames, filenames in os.walk(path):
        for d in dirnames:
            if "img_align_cel" in d:
                nested = os.path.join(dirpath, d, d)
                if os.path.isdir(nested):
                    return nested
                return os.path.join(dirpath, d)
    return None

LABEL_CSV = _find_csv(".")
IMG_DIR = _find_img(".")
if not IMG_DIR:
    IMG_DIR = os.path.join("celebA_dataset", "img_align_cel")
IMAGE_SIZE = 224


def compute_crop_params(bx, by, bw, bh):
    crop_scale = 2.14
    crop_ar = 1.22
    crop_w = np.maximum(10, np.floor(bw * crop_scale)).astype(np.float32)
    crop_h = np.maximum(10, np.floor(crop_w * crop_ar)).astype(np.float32)
    crop_x = np.floor(bx + bw / 2.0 - crop_w / 2.0).astype(np.float32)
    crop_y = np.floor(by + bh * 0.4 - crop_h * 0.51).astype(np.float32)
    return crop_x, crop_y, crop_w, crop_h


def main():
    df = pd.read_csv(LABEL_CSV)
    df = df[df["partition"] == 2].reset_index(drop=True)

    bx = df["x_1"].to_numpy(np.float32)
    by = df["y_1"].to_numpy(np.float32)
    bw = df["width"].to_numpy(np.float32)
    bh = df["height"].to_numpy(np.float32)

    crop_x, crop_y, crop_w, crop_h = compute_crop_params(bx, by, bw, bh)

    lm_raw = df[[
        "lefteye_x", "lefteye_y",
        "righteye_x", "righteye_y",
        "nose_x", "nose_y",
        "leftmouth_x", "leftmouth_y",
        "rightmouth_x", "rightmouth_y",
    ]].to_numpy(np.float32)

    # OLD: zero-pad logic → coordinates clipped to [0,1]
    lm_old = np.empty_like(lm_raw)
    lm_old[:, 0::2] = np.clip((lm_raw[:, 0::2] - crop_x[:, None]) / crop_w[:, None], 0.0, 1.0)
    lm_old[:, 1::2] = np.clip((lm_raw[:, 1::2] - crop_y[:, None]) / crop_h[:, None], 0.0, 1.0)

    # NEW: BORDER_REPLICATE logic → no clipping
    lm_new = np.empty_like(lm_raw)
    lm_new[:, 0::2] = (lm_raw[:, 0::2] - crop_x[:, None]) / crop_w[:, None]
    lm_new[:, 1::2] = (lm_raw[:, 1::2] - crop_y[:, None]) / crop_h[:, None]

    iod_old = np.hypot(lm_old[:, 0] - lm_old[:, 2], lm_old[:, 1] - lm_old[:, 3])
    iod_new = np.hypot(lm_new[:, 0] - lm_new[:, 2], lm_new[:, 1] - lm_new[:, 3])

    clipped_old = (lm_old == 0.0) | (lm_old == 1.0)
    out_new     = (lm_new < 0.0) | (lm_new > 1.0)

    print(f"partition2_samples={len(df)}")
    print()
    print("=" * 60)
    print("  IOD (interocular distance) comparison")
    print("=" * 60)
    print(f"  OLD (zero-pad):  mean={iod_old.mean():.6f}  median={np.median(iod_old):.6f}  p5={np.percentile(iod_old, 5):.6f}")
    print(f"  NEW (replicate): mean={iod_new.mean():.6f}  median={np.median(iod_new):.6f}  p5={np.percentile(iod_new, 5):.6f}")
    print()
    print("  IOD < threshold percentage:")
    for t in [0.01, 0.02, 0.03, 0.05]:
        print(f"    IOD < {t:.2f}: OLD={100*(iod_old<t).mean():.4f}%  NEW={100*(iod_new<t).mean():.4f}%  delta={100*((iod_new<t).mean()-(iod_old<t).mean()):+.4f}%")
    print()
    print("=" * 60)
    print("  Landmark coordinate clipping")
    print("=" * 60)
    print(f"  OLD coords clipped to 0 or 1:  {100*clipped_old.mean():.4f}%  ({100*clipped_old.any(axis=1).mean():.4f}% samples)")
    print(f"  NEW coords out of [0,1]:       {100*out_new.mean():.4f}%  ({100*out_new.any(axis=1).mean():.4f}% samples)")
    print()
    low  = (lm_new < 0.0).any(axis=1).sum()
    high = (lm_new > 1.0).any(axis=1).sum()
    any_out = ((lm_new < 0.0) | (lm_new > 1.0)).any(axis=1).sum()
    print(f"  NEW samples with coords < 0: {low:,}")
    print(f"  NEW samples with coords > 1: {high:,}")
    print(f"  NEW samples out of [0,1]:   {any_out:,}")
    print()
    print("  Note: Out-of-bounds coords are clipped by PyTorch normalize().")
    print("  BORDER_REPLICATE eliminates black padding entirely.")


if __name__ == "__main__":
    main()
