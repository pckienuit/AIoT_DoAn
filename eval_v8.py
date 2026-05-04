"""
eval_v8.py — Đánh giá model train_v8.py (MobileNetV2 + BORDER_REPLICATE crop)
trên test set (partition=2).

Metrics:
  - Classification accuracy (sigmoid threshold 0.5)
  - NME  (Normalized Mean Error) — mean(|predicted - gt|) / interocular_distance
  - Per-landmark MAE (pixel) để biết điểm nào yếu
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_v8 import FaceDetectMultiTask, CelebADataset, IMAGE_SIZE, IMG_DIR, LABEL_CSV

MODEL_PATH = "face_detect_model_vps_finetune_v9.pth"
if len(sys.argv) > 1:
    MODEL_PATH = sys.argv[1]
BATCH_SIZE = 64
NUM_WORKERS = 0

LANDMARK_NAMES = ["L.Eye", "R.Eye", "Nose", "L.Mouth", "R.Mouth"]
NME_MIN_IOD_NORM = 0.05
LEYE_IDX = (0, 1)
REYE_IDX = (2, 3)


def interocular_distance(lm_gt: np.ndarray) -> float:
    lx, ly = lm_gt[LEYE_IDX[0]], lm_gt[LEYE_IDX[1]]
    rx, ry = lm_gt[REYE_IDX[0]], lm_gt[REYE_IDX[1]]
    return float(np.sqrt((rx - lx) ** 2 + (ry - ly) ** 2))


def denormalize_to_input_px(lm_norm: np.ndarray) -> np.ndarray:
    lm_px = np.clip(lm_norm, 0.0, 1.0).copy()
    lm_px[0::2] *= IMAGE_SIZE
    lm_px[1::2] *= IMAGE_SIZE
    return lm_px


def is_clean_gt_sample(lm_gt_norm: np.ndarray) -> bool:
    if np.any((lm_gt_norm <= 0.0) | (lm_gt_norm >= 1.0)):
        return False
    iod_norm = float(
        np.hypot(
            lm_gt_norm[LEYE_IDX[0]] - lm_gt_norm[REYE_IDX[0]],
            lm_gt_norm[LEYE_IDX[1]] - lm_gt_norm[REYE_IDX[1]],
        )
    )
    return iod_norm >= NME_MIN_IOD_NORM


def evaluate(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = FaceDetectMultiTask()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        val = checkpoint.get('best_val_loss', '?')
        print(f"[OK] Loaded checkpoint | epoch={epoch} | val_loss={val}")
    else:
        model.load_state_dict(checkpoint)
        print(f"[OK] Loaded weights-only | {model_path}")
    model.to(device)
    model.eval()

    test_dataset = CelebADataset(LABEL_CSV, IMG_DIR, partition=2, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)
    print(f"Test: {len(test_dataset)} samples | {len(test_loader)} batches\n")

    correct_class = 0
    total_samples = 0
    positive_samples = 0
    clean_nme_samples = 0
    per_lm_errors = np.zeros(5)
    nme_all_sum = 0.0
    nme_all_count = 0
    nme_sum = 0.0
    nme_count = 0

    is_kaggle = os.path.exists("/kaggle/input")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating", disable=is_kaggle)
        for images, targets in pbar:
            class_labels, _, landmarks_gt = targets

            images = images.to(device)
            class_labels = class_labels.to(device)
            landmarks_gt = landmarks_gt.to(device)

            class_out, _, landmark_out = model(images)

            preds = (torch.sigmoid(class_out) >= 0.5).float()
            correct_class += (preds == class_labels).sum().item()
            total_samples += class_labels.size(0)

            lm_pred_np = landmark_out.cpu().numpy()
            lm_gt_np = landmarks_gt.cpu().numpy()
            cls_np = class_labels.cpu().numpy().reshape(-1)

            for i in range(lm_pred_np.shape[0]):
                if cls_np[i] < 0.5:
                    continue

                positive_samples += 1
                pred_px = denormalize_to_input_px(lm_pred_np[i])
                gt_px = denormalize_to_input_px(lm_gt_np[i])

                for j in range(5):
                    dx = pred_px[j*2] - gt_px[j*2]
                    dy = pred_px[j*2 + 1] - gt_px[j*2 + 1]
                    per_lm_errors[j] += np.sqrt(dx**2 + dy**2)

                iod = interocular_distance(gt_px)
                if iod > 1.0:
                    mean_err_px = np.mean([
                        np.sqrt((pred_px[k*2] - gt_px[k*2])**2 +
                                (pred_px[k*2+1] - gt_px[k*2+1])**2)
                        for k in range(5)
                    ])
                    nme_all_sum += mean_err_px / iod
                    nme_all_count += 1

                gt_norm = np.clip(lm_gt_np[i], 0.0, 1.0)
                if is_clean_gt_sample(gt_norm):
                    clean_nme_samples += 1
                    mean_err_px = np.mean([
                        np.sqrt((pred_px[k*2] - gt_px[k*2])**2 +
                                (pred_px[k*2+1] - gt_px[k*2+1])**2)
                        for k in range(5)
                    ])
                    nme_sum += mean_err_px / iod
                    nme_count += 1

    acc = correct_class / total_samples * 100
    nme_all = (nme_all_sum / nme_all_count) * 100 if nme_all_count else float("nan")
    nme = (nme_sum / nme_count) * 100 if nme_count else float("nan")
    per_lm_mae = per_lm_errors / positive_samples if positive_samples else np.zeros(5)

    print("=" * 55)
    print("         TEST SET EVALUATION RESULTS (v8)")
    print("=" * 55)
    print(f"  Total test samples     : {total_samples:,}")
    print(f"  Positive (landmark)    : {positive_samples:,}")
    print(f"  Clean NME samples      : {clean_nme_samples:,}")
    print(f"  Classification Accuracy: {acc:.2f}%")
    print(f"  NME (%% interocular) all+: {nme_all:.2f}%  (all positive)")
    print(f"  NME (%% interocular) clean: {nme:.2f}%  (clean subset)")
    print("-" * 55)
    print("  Per-Landmark Error (pixel Euclidean):")
    for j, name in enumerate(LANDMARK_NAMES):
        bar = "#" * int(per_lm_mae[j] / 2)
        print(f"    {name:<10}: {per_lm_mae[j]:6.2f} px  {bar}")
    print("=" * 55)

    if nme < 4:
        grade = "[GOOD] Excellent"
    elif nme < 7:
        grade = "[OK] Acceptable"
    elif nme < 10:
        grade = "[WARN] Needs improvement"
    else:
        grade = "[BAD] Poor - needs augmentation / more epochs"
    print(f"\n  NME grade (WFLW/300W standard):")
    print(f"  NME = {nme:.2f}% -> {grade}")


if __name__ == "__main__":
    evaluate(MODEL_PATH)
