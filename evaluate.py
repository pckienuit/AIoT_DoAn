"""
evaluate.py — Đánh giá mô hình FaceDetectMultiTask trên test set (partition=2).

Metrics:
  - Classification accuracy (sigmoid threshold 0.5)
  - NME  (Normalized Mean Error) — tiêu chuẩn quốc tế cho facial landmark
    NME = mean(|predicted - gt|) / interocular_distance
  - Per-landmark MAE (pixel) để biết điểm nào mô hình còn yếu
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import FaceDetectMultiTask, CelebADataset, IMAGE_SIZE, IMG_DIR, LABEL_CSV, IMG_W, IMG_H

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH  = "face_detect_model_withval.pth"
BATCH_SIZE  = 64
NUM_WORKERS = 0

LANDMARK_NAMES = ["L.Eye", "R.Eye", "Nose", "L.Mouth", "R.Mouth"]

# Interocular distance dùng để normalize NME:
# = khoảng cách giữa 2 mắt (lefteye, righteye) → index 0,1 và 2,3 trong lm vector
LEYE_IDX  = (0, 1)   # lefteye  (x, y)
REYE_IDX  = (2, 3)   # righteye (x, y)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def interocular_distance(lm_gt: np.ndarray) -> float:
    """
    Tính khoảng cách 2 mắt (pixel) từ GT landmarks đã denormalize.
    lm_gt shape: (10,) — đã scale về pixel
    """
    lx, ly = lm_gt[LEYE_IDX[0]], lm_gt[LEYE_IDX[1]]
    rx, ry = lm_gt[REYE_IDX[0]], lm_gt[REYE_IDX[1]]
    return float(np.sqrt((rx - lx) ** 2 + (ry - ly) ** 2))


def denormalize(lm_norm: np.ndarray) -> np.ndarray:
    """
    Chuyển landmarks từ [0,1] → pixel.
    lm_norm shape: (10,) — x ở chỉ số chẵn, y ở chỉ số lẻ
    """
    lm_px = lm_norm.copy()
    lm_px[0::2] *= IMG_W
    lm_px[1::2] *= IMG_H
    return lm_px


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    # Load model
    model = FaceDetectMultiTask()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Đã load model từ '{model_path}'")

    # Test dataset (partition=2)
    test_dataset = CelebADataset(LABEL_CSV, IMG_DIR, partition=2)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)
    print(f"Test set: {len(test_dataset)} samples | {len(test_loader)} batches\n")

    # Accumulators
    correct_class   = 0
    total_samples   = 0
    per_lm_errors   = np.zeros(5)   # MAE theo pixel cho từng landmark
    nme_sum         = 0.0
    nme_count       = 0

    with torch.no_grad():
        for images, (class_labels, landmarks_gt) in tqdm(test_loader, desc="Evaluating"):
            images       = images.to(device)
            class_labels = class_labels.to(device)   # shape (B,1)
            landmarks_gt = landmarks_gt.to(device)    # shape (B,10)

            class_out, _, landmark_out = model(images)

            # ---- Classification accuracy ----
            preds = (torch.sigmoid(class_out) >= 0.5).float()
            correct_class += (preds == class_labels).sum().item()
            total_samples += class_labels.size(0)

            # ---- Landmark metrics (per sample) ----
            lm_pred_np = landmark_out.cpu().numpy()   # (B,10)
            lm_gt_np   = landmarks_gt.cpu().numpy()   # (B,10)

            for i in range(lm_pred_np.shape[0]):
                pred_px = denormalize(lm_pred_np[i])
                gt_px   = denormalize(lm_gt_np[i])

                # Per-landmark absolute error (Euclidean distance, pixel)
                for j in range(5):
                    dx = pred_px[j*2]     - gt_px[j*2]
                    dy = pred_px[j*2 + 1] - gt_px[j*2 + 1]
                    per_lm_errors[j] += np.sqrt(dx**2 + dy**2)

                # NME
                iod = interocular_distance(gt_px)
                if iod > 1e-6:   # tránh chia 0
                    mean_err_px = np.mean([
                        np.sqrt((pred_px[j*2] - gt_px[j*2])**2 +
                                (pred_px[j*2+1] - gt_px[j*2+1])**2)
                        for j in range(5)
                    ])
                    nme_sum   += mean_err_px / iod
                    nme_count += 1

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    acc = correct_class / total_samples * 100
    nme = (nme_sum / nme_count) * 100 if nme_count else float("nan")
    per_lm_mae = per_lm_errors / total_samples

    print("=" * 55)
    print("         KẾT QUẢ ĐÁNH GIÁ TRÊN TEST SET")
    print("=" * 55)
    print(f"  Tổng mẫu test         : {total_samples:,}")
    print(f"  Classification Accuracy: {acc:.2f}%")
    print(f"  NME (% interocular)   : {nme:.2f}%")
    print("-" * 55)
    print("  Per-Landmark Error (pixel Euclidean):")
    for j, name in enumerate(LANDMARK_NAMES):
        bar = "█" * int(per_lm_mae[j] / 2)
        print(f"    {name:<10}: {per_lm_mae[j]:6.2f} px  {bar}")
    print("=" * 55)

    # Đánh giá chất lượng NME
    print("\n  Thang đánh giá NME (tiêu chuẩn WFLW/300W):")
    if nme < 4:
        grade = "🟢 Tốt"
    elif nme < 7:
        grade = "🟡 Chấp nhận được"
    elif nme < 10:
        grade = "🟠 Cần cải thiện"
    else:
        grade = "🔴 Kém — cần augmentation / thêm epochs"
    print(f"  NME = {nme:.2f}% → {grade}")
    print()


if __name__ == "__main__":
    evaluate(MODEL_PATH)
