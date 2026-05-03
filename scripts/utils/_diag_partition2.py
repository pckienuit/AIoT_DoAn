import numpy as np
import pandas as pd

from train import LABEL_CSV, IMAGE_SIZE


def compute_crop_params(bx: np.ndarray, by: np.ndarray, bw: np.ndarray, bh: np.ndarray):
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

    lm_raw = df[
        [
            "lefteye_x", "lefteye_y",
            "righteye_x", "righteye_y",
            "nose_x", "nose_y",
            "leftmouth_x", "leftmouth_y",
            "rightmouth_x", "rightmouth_y",
        ]
    ].to_numpy(np.float32)

    lm_norm = np.empty_like(lm_raw)
    lm_norm[:, 0::2] = np.clip((lm_raw[:, 0::2] - crop_x[:, None]) / crop_w[:, None], 0.0, 1.0)
    lm_norm[:, 1::2] = np.clip((lm_raw[:, 1::2] - crop_y[:, None]) / crop_h[:, None], 0.0, 1.0)

    iod_norm = np.hypot(lm_norm[:, 0] - lm_norm[:, 2], lm_norm[:, 1] - lm_norm[:, 3])
    iod_px = iod_norm * IMAGE_SIZE

    thresholds = [0.01, 0.02, 0.03, 0.05]
    clipped_mask = (lm_norm == 0.0) | (lm_norm == 1.0)

    print(f"partition2_samples={len(df)}")
    print(
        "iod_norm_stats="
        f"min:{iod_norm.min():.6f}, mean:{iod_norm.mean():.6f}, median:{np.median(iod_norm):.6f}, "
        f"p5:{np.percentile(iod_norm, 5):.6f}, p95:{np.percentile(iod_norm, 95):.6f}"
    )
    print(
        "iod_px_stats="
        f"min:{iod_px.min():.3f}, mean:{iod_px.mean():.3f}, median:{np.median(iod_px):.3f}"
    )
    for t in thresholds:
        print(f"iod_lt_{t:.2f}_pct={100.0 * np.mean(iod_norm < t):.4f}")

    print(f"coords_clipped_to_0_or_1_pct={100.0 * clipped_mask.mean():.4f}")
    print(f"samples_with_any_clipped_coord_pct={100.0 * np.mean(clipped_mask.any(axis=1)):.4f}")


if __name__ == "__main__":
    main()
