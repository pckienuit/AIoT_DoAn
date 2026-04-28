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
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import FaceDetectMultiTask, CelebADataset, IMAGE_SIZE, IMG_DIR, LABEL_CSV

import sys

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH  = "face_detect_model_withval8.pth"
if len(sys.argv) > 1:
    MODEL_PATH = sys.argv[1]
BATCH_SIZE  = 64
NUM_WORKERS = 0

LANDMARK_NAMES = ["L.Eye", "R.Eye", "Nose", "L.Mouth", "R.Mouth"]
NME_MIN_IOD_NORM = 0.05

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


def denormalize_to_input_px(lm_norm: np.ndarray) -> np.ndarray:
    """
    Chuyển landmarks từ [0,1] → pixel trong hệ tọa độ input IMAGE_SIZE x IMAGE_SIZE.
    lm_norm shape: (10,) — x ở chỉ số chẵn, y ở chỉ số lẻ.
    """
    lm_px = np.clip(lm_norm, 0.0, 1.0).copy()
    lm_px[0::2] *= IMAGE_SIZE
    lm_px[1::2] *= IMAGE_SIZE
    return lm_px


def is_clean_gt_sample(lm_gt_norm: np.ndarray) -> bool:
    """GT hợp lệ để tính NME: không bị clip biên và có IOD đủ lớn."""
    if np.any((lm_gt_norm <= 0.0) | (lm_gt_norm >= 1.0)):
        return False
    iod_norm = float(
        np.hypot(
            lm_gt_norm[LEYE_IDX[0]] - lm_gt_norm[REYE_IDX[0]],
            lm_gt_norm[LEYE_IDX[1]] - lm_gt_norm[REYE_IDX[1]],
        )
    )
    return iod_norm >= NME_MIN_IOD_NORM


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
    positive_samples = 0
    clean_nme_samples = 0
    per_lm_errors   = np.zeros(5)   # MAE theo pixel cho từng landmark
    nme_all_sum     = 0.0
    nme_all_count   = 0
    nme_sum         = 0.0
    nme_count       = 0

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            # Dataset hiện tại trả về (class_labels, bboxes, landmarks).
            # Giữ tương thích nếu gặp format cũ chỉ có (class_labels, landmarks).
            if isinstance(targets, (list, tuple)) and len(targets) == 3:
                class_labels, _, landmarks_gt = targets
            elif isinstance(targets, (list, tuple)) and len(targets) == 2:
                class_labels, landmarks_gt = targets
            else:
                raise ValueError(
                    f"Unsupported target format from DataLoader: type={type(targets)}"
                )

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
            cls_np     = class_labels.cpu().numpy().reshape(-1)

            for i in range(lm_pred_np.shape[0]):
                # Chỉ đánh giá landmark/NME trên positive samples (có mặt thật).
                if cls_np[i] < 0.5:
                    continue

                positive_samples += 1
                pred_px = denormalize_to_input_px(lm_pred_np[i])
                gt_px   = denormalize_to_input_px(lm_gt_np[i])

                # Per-landmark absolute error (Euclidean distance, pixel)
                for j in range(5):
                    dx = pred_px[j*2]     - gt_px[j*2]
                    dy = pred_px[j*2 + 1] - gt_px[j*2 + 1]
                    per_lm_errors[j] += np.sqrt(dx**2 + dy**2)

                # NME all-positive (chỉ bỏ sample có iod quá nhỏ để tránh chia 0).
                iod = interocular_distance(gt_px)
                if iod > 1.0:
                    mean_err_px = np.mean([
                        np.sqrt((pred_px[j*2] - gt_px[j*2])**2 +
                                (pred_px[j*2+1] - gt_px[j*2+1])**2)
                        for j in range(5)
                    ])
                    nme_all_sum += mean_err_px / iod
                    nme_all_count += 1

                # NME chỉ tính trên mẫu GT sạch để tránh chia bởi IOD gần 0.
                gt_norm = np.clip(lm_gt_np[i], 0.0, 1.0)
                if is_clean_gt_sample(gt_norm):
                    clean_nme_samples += 1
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
    nme_all = (nme_all_sum / nme_all_count) * 100 if nme_all_count else float("nan")
    nme = (nme_sum / nme_count) * 100 if nme_count else float("nan")
    per_lm_mae = per_lm_errors / positive_samples if positive_samples else np.zeros(5)

    print("=" * 55)
    print("         KẾT QUẢ ĐÁNH GIÁ TRÊN TEST SET")
    print("=" * 55)
    print(f"  Tổng mẫu test         : {total_samples:,}")
    print(f"  Mẫu positive (landmark): {positive_samples:,}")
    print(f"  Mẫu clean cho NME      : {clean_nme_samples:,}")
    print(f"  Classification Accuracy: {acc:.2f}%")
    print(f"  NME (% interocular) all+: {nme_all:.2f}%  (all positive)")
    print(f"  NME (% interocular) clean: {nme:.2f}%  (clean subset)")
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
