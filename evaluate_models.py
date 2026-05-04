"""
evaluate_models.py — Đánh giá so sánh tất cả các model checkpoint trong thư mục.

Metrics đánh giá:
  - Classification : Accuracy, Precision, Recall, F1, AUC-ROC
  - Bounding Box  : MSE, MAE
  - Landmarks     : MSE, MAE, NME (Normalized Mean Error)
  - Combined Loss : tổng hợp từ train_v9.py

Kết quả được lưu vào:
  results/evaluation_results.csv
  results/evaluation_metrics.png

Usage:
  python evaluate_models.py
  python evaluate_models.py --checkpoint_dir models/checkpoints --max_samples 5000
"""

import os
import sys
import math
import argparse
import itertools
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from tqdm import tqdm
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not found — plotting disabled")

# ─── Config ───────────────────────────────────────────────────────────────────
IMAGE_SIZE    = 224
LABEL_CSV_ENV = os.environ.get("LABEL_CSV", None)
IMG_DIR_ENV   = os.environ.get("IMG_DIR",   None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")

# ─── Dataset paths (tái sử dụng logic từ train_v9.py) ───────────────────────
def _find_dataset_paths(root="."):
    label_csv = img_dir = None
    for dirpath, _, filenames in os.walk(root):
        if label_csv is None and "labels.csv" in filenames:
            label_csv = os.path.join(dirpath, "labels.csv")
        if img_dir is None and os.path.basename(dirpath).lower() == "img_align_celeba":
            img_dir = dirpath
        if label_csv and img_dir:
            break
    if img_dir:
        for nested in ("img_align_celebA", "img_align_celebra"):
            n = os.path.join(img_dir, nested)
            if os.path.isdir(n):
                img_dir = n
                break
    return label_csv, img_dir

if LABEL_CSV_ENV and IMG_DIR_ENV:
    LABEL_CSV, IMG_DIR = LABEL_CSV_ENV, IMG_DIR_ENV
elif os.path.exists("/kaggle/input"):
    LABEL_CSV, IMG_DIR = _find_dataset_paths("/kaggle/input")
else:
    LABEL_CSV, IMG_DIR = _find_dataset_paths(".")
    if not LABEL_CSV or not IMG_DIR:
        for csv_p, img_p in [
            ("/root/labels.csv",        "/root/img_align_celebA/img_align_celebA"),
            ("/data/labels.csv",        "/data/img_align_celebA/img_align_celebA"),
        ]:
            if os.path.exists(csv_p):
                LABEL_CSV, IMG_DIR = csv_p, img_p
                break
        if not LABEL_CSV:
            LABEL_CSV = "labels.csv"
            IMG_DIR   = "celebA_dataset/img_align_celebra"

print(f"[Dataset] labels={LABEL_CSV}")
print(f"[Dataset] img_dir={IMG_DIR}")


# ─── Loss helpers (đồng nhất train_v9.py) ────────────────────────────────────
def wing_loss(pred, target, w=10.0, eps=2.0):
    SCALE = 224.0
    x = (pred * SCALE - target * SCALE).abs()
    C = w - w * math.log(1.0 + w / eps)
    return torch.where(x < w, w * torch.log(1.0 + x / eps), x - C).mean() / SCALE


class FocalLandmarkLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma, self.scale = alpha, gamma, 224.0

    def forward(self, pred, target):
        diff = (pred - target).abs() * self.scale
        pt = torch.exp(-diff)
        return (self.alpha * (1 - pt) ** self.gamma * diff).mean() / self.scale


# ─── Model (tái sử dụng từ train_v9.py) ─────────────────────────────────────
class FaceDetectMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.backbone = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Classification head
        self.class_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        # Bounding box head
        self.bbox_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
        )
        # Landmark head (MLP)
        self.landmark_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        return (
            self.class_head(f).squeeze(-1),
            self.bbox_head(f),
            self.landmark_head(f),
        )


def load_model_weights(model, ckpt_path):
    """Load weights với tự động tương thích ngược cho checkpoint cũ."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        sd = ckpt

    # Load thử — nếu lỗi shape thì thử bỏ qua mismatch
    try:
        model.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        print(f"  [!] strict=False failed: {e}")
        # Backward compat: landmark_head cũ là Linear(1280, 10)
        if "landmark_head.0.weight" not in sd and "landmark_head.weight" in sd:
            print("  [~] Replacing landmark_head with old Linear(1280,10) layout")
            model.landmark_head = nn.Linear(1280, 10)
            model.load_state_dict(sd, strict=False)
    return model


# ─── Validation Dataset ───────────────────────────────────────────────────────
class CelebAValDataset(Dataset):
    """Validation set: positive + negative samples (không augment)."""

    def __init__(self, csv_file, img_dir, partition=1, neg_ratio=0.3):
        df = pd.read_csv(csv_file)
        df = df[df["partition"] == partition].reset_index(drop=True)
        self.data     = df
        self.img_dir  = img_dir
        self.neg_ratio = neg_ratio

    def __len__(self):
        return len(self.data)

    def _generate_negative_crop(self, image_full, bbox_raw):
        H, W = image_full.shape[:2]
        for _ in range(10):
            crop_size = np.random.randint(40, max(40, min(W, H) // 2))
            cx, cy    = np.random.randint(0, max(0, W - crop_size)), \
                        np.random.randint(0, max(0, H - crop_size))
            ix1, iy1  = max(cx, int(bbox_raw[0])), max(cy, int(bbox_raw[1]))
            ix2, iy2  = min(cx + crop_size, int(bbox_raw[0]) + int(bbox_raw[2])), \
                        min(cy + crop_size, int(bbox_raw[1]) + int(bbox_raw[3]))
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter < (crop_size * crop_size * 0.1):
                return image_full[cy:cy + crop_size, cx:cx + crop_size]
        return image_full[0:30, 0:30]

    def _make_crop(self, image_full, bx, by, bw, bh):
        CROP_SCALE, CROP_AR = 2.14, 1.22
        crop_w = max(10, int(bw * CROP_SCALE))
        crop_h = max(10, int(crop_w * CROP_AR))
        crop_x = int(bx + bw / 2 - crop_w / 2)
        crop_y = int(by + bh * 0.4 - crop_h * 0.51)
        H, W   = image_full.shape[:2]

        pl, pt = max(0, -crop_x), max(0, -crop_y)
        pr, pb = max(0, (crop_x + crop_w) - W), max(0, (crop_y + crop_h) - H)
        if pl or pt or pr or pb:
            image_full = cv2.copyMakeBorder(
                image_full, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
            crop_x += pl
            crop_y += pt

        return image_full[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], crop_x - pl, crop_y - pt, crop_w, crop_h

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        image_full = cv2.imread(img_path)
        if image_full is None:
            return self.__getitem__((idx + 1) % len(self))
        image_full = cv2.cvtColor(image_full, cv2.COLOR_BGR2RGB)
        bbox_raw = row[["x_1", "y_1", "width", "height"]].values.astype(np.float32)

        is_neg = np.random.random() < self.neg_ratio
        if is_neg:
            crop = self._generate_negative_crop(image_full, bbox_raw)
            image = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            return (
                torch.tensor(image),
                (torch.tensor([0.0]), torch.zeros(4), torch.zeros(10))
            )

        bx, by, bw, bh = bbox_raw
        face_crop, crop_x, crop_y, crop_w, crop_h = self._make_crop(
            image_full, bx, by, bw, bh)
        image = cv2.resize(face_crop, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0

        bbox = np.array([
            np.clip((bx - crop_x) / crop_w, 0.0, 1.0),
            np.clip((by - crop_y) / crop_h, 0.0, 1.0),
            np.clip(bw / crop_w, 0.0, 1.0),
            np.clip(bh / crop_h, 0.0, 1.0),
        ], dtype=np.float32)

        landmark_raw = row[["lefteye_x",  "lefteye_y",
                            "righteye_x", "righteye_y",
                            "nose_x",     "nose_y",
                            "leftmouth_x","leftmouth_y",
                            "rightmouth_x","rightmouth_y"]].values.astype(np.float32)
        landmarks = np.zeros(10, dtype=np.float32)
        landmarks[0::2] = np.clip((landmark_raw[0::2] - crop_x) / crop_w, 0.0, 1.0)
        landmarks[1::2] = np.clip((landmark_raw[1::2] - crop_y) / crop_h, 0.0, 1.0)

        image = np.transpose(image, (2, 0, 1))
        return (
            torch.tensor(image),
            (torch.tensor([1.0]), torch.tensor(bbox), torch.tensor(landmarks))
        )


# ─── Evaluation per model ──────────────────────────────────────────────────────
def evaluate_model(model, dataloader, device):
    """Chạy đánh giá trên validation set, trả về dict metrics."""
    model.eval()

    all_cls_gt  = []
    all_cls_pr  = []
    all_cls_prb = []
    all_bbox_gt = []
    all_bbox_pr = []
    all_lm_gt   = []
    all_lm_pr   = []
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_lm_loss   = 0.0
    n_pos = 0

    cls_fn  = nn.BCEWithLogitsLoss()
    bbox_fn = nn.L1Loss()
    lm_wing = lambda p, t: wing_loss(p, t)
    lm_focal = FocalLandmarkLoss()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            imgs = images.to(device)
            cls_t, bbox_t, lm_t = targets[0].to(device), targets[1].to(device), targets[2].to(device)

            cls_p, bbox_p, lm_p = model(imgs)

            # Classification targets: [B,1] -> squeeze to [B]
            cls_t_1d = cls_t.squeeze(-1)  # [B]

            # Classification
            all_cls_gt.append(cls_t_1d.cpu().numpy())
            all_cls_pr.append((torch.sigmoid(cls_p) > 0.5).cpu().numpy())
            all_cls_prb.append(torch.sigmoid(cls_p).cpu().numpy())

            # Positive samples only for bbox/landmark
            pos_mask = cls_t_1d > 0.5
            if pos_mask.any():
                n_pos += pos_mask.sum().item()
                all_bbox_gt.append(bbox_t[pos_mask].cpu().numpy())
                all_bbox_pr.append(bbox_p[pos_mask].cpu().numpy())
                all_lm_gt.append(lm_t[pos_mask].cpu().numpy())
                all_lm_pr.append(lm_p[pos_mask].cpu().numpy())
                total_cls_loss  += cls_fn(cls_p[pos_mask], cls_t_1d[pos_mask]).item() * pos_mask.sum().item()
                total_bbox_loss += bbox_fn(bbox_p[pos_mask], bbox_t[pos_mask]).item() * pos_mask.sum().item()
                total_lm_loss   += (lm_wing(lm_p[pos_mask], lm_t[pos_mask]) +
                                    lm_focal(lm_p[pos_mask], lm_t[pos_mask])).item() * pos_mask.sum().item()
            else:
                total_cls_loss += cls_fn(cls_p, cls_t_1d).item() * len(cls_t_1d)

    # Flatten arrays
    cls_gt  = np.concatenate(all_cls_gt)
    cls_pr  = np.concatenate(all_cls_pr)
    cls_prb = np.concatenate(all_cls_prb)
    n_total = len(cls_gt)
    n_neg   = n_total - n_pos

    bbox_gt_arr = np.concatenate(all_bbox_gt) if all_bbox_gt else np.zeros((0, 4))
    bbox_pr_arr = np.concatenate(all_bbox_pr) if all_bbox_pr else np.zeros((0, 4))
    lm_gt_arr   = np.concatenate(all_lm_gt)   if all_lm_gt   else np.zeros((0, 10))
    lm_pr_arr   = np.concatenate(all_lm_pr)   if all_lm_pr   else np.zeros((0, 10))

    # Classification metrics
    acc  = accuracy_score(cls_gt, cls_pr)
    prec = precision_score(cls_gt, cls_pr, zero_division=0)
    rec  = recall_score(cls_gt, cls_pr, zero_division=0)
    f1   = f1_score(cls_gt, cls_pr, zero_division=0)
    try:
        auc = roc_auc_score(cls_gt, cls_prb)
    except ValueError:
        auc = 0.0
    try:
        ap = average_precision_score(cls_gt, cls_prb)
    except ValueError:
        ap = 0.0

    # Bounding box metrics (positive samples only)
    if n_pos > 0:
        bbox_mse = np.mean((bbox_gt_arr - bbox_pr_arr) ** 2)
        bbox_mae = np.mean(np.abs(bbox_gt_arr - bbox_pr_arr))
    else:
        bbox_mse = bbox_mae = float("nan")

    # Landmark metrics (positive samples only)
    if n_pos > 0:
        lm_mse = np.mean((lm_gt_arr - lm_pr_arr) ** 2)
        lm_mae = np.mean(np.abs(lm_gt_arr - lm_pr_arr))
        # NME: normalized by bounding box size (diagonal of bbox)
        bbox_sizes = np.sqrt(bbox_gt_arr[:, 2] ** 2 + bbox_gt_arr[:, 3] ** 2 + 1e-6)  # [N,]
        # Average per-landmark NME
        nme_per_lm = np.sqrt(np.sum((lm_gt_arr - lm_pr_arr) ** 2, axis=1)) / bbox_sizes  # [N,]
        nme = np.mean(nme_per_lm) * 100  # as percentage
        # Per-landmark MAE
        per_lm_mae = np.mean(np.abs(lm_gt_arr - lm_pr_arr), axis=0)  # [10,]
    else:
        lm_mse = lm_mae = nme = float("nan")
        per_lm_mae = np.zeros(10)

    avg_cls_loss   = total_cls_loss   / n_total
    avg_bbox_loss  = total_bbox_loss / max(n_pos, 1)
    avg_lm_loss    = total_lm_loss    / max(n_pos, 1)
    combined_loss  = avg_cls_loss + 5.0 * avg_bbox_loss + 30.0 * avg_lm_loss

    return {
        "n_samples":       n_total,
        "n_positive":      n_pos,
        "n_negative":      n_neg,
        "cls_accuracy":    acc,
        "cls_precision":   prec,
        "cls_recall":      rec,
        "cls_f1":          f1,
        "cls_auc_roc":     auc,
        "cls_avg_precision": ap,
        "cls_loss":        avg_cls_loss,
        "bbox_mse":        bbox_mse,
        "bbox_mae":        bbox_mae,
        "bbox_loss":       avg_bbox_loss,
        "landmark_mse":    lm_mse,
        "landmark_mae":    lm_mae,
        "landmark_nme":    nme,
        "landmark_loss":   avg_lm_loss,
        "combined_loss":   combined_loss,
        "per_landmark_mae": per_lm_mae,
    }


# ─── Scan checkpoints ─────────────────────────────────────────────────────────
def find_checkpoints(root_dir):
    """Trả về list các đường dẫn .pth, loại trừ weights-only files."""
    checkpoints = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".pth") and "weights" not in f:
                checkpoints.append(os.path.join(dirpath, f))
    return sorted(set(checkpoints))


# ─── Plot comparison charts ───────────────────────────────────────────────────
def plot_comparison(results_df, output_path):
    """Vẽ các biểu đồ so sánh giữa các model."""
    if results_df.empty:
        return

    LANDMARK_NAMES = ["L.Eye", "R.Eye", "Nose", "L.Mouth", "R.Mouth"]
    models  = results_df["model_name"].tolist()
    short   = [m.replace("face_detect_model_vps_finetune_", "").replace(".pth", "").replace("_", "\n") for m in models]
    colors  = plt.cm.tab10(np.linspace(0, 1, len(models)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Model Comparison — Validation Set", fontsize=16, fontweight="bold")

    def bar(ax, values, title, ylabel, higher_better=True, fmt=".3f"):
        bars = ax.bar(short, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=0, labelsize=7)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:{fmt}}", ha="center", va="bottom", fontsize=7, rotation=0)

    # Classification group
    bar(axes[0, 0], results_df["cls_accuracy"].values,    "Classification Accuracy",  "Accuracy",  fmt=".3f")
    bar(axes[0, 1], results_df["cls_f1"].values,          "Classification F1 Score",    "F1",        fmt=".3f")
    bar(axes[0, 2], results_df["cls_auc_roc"].values,     "Classification AUC-ROC",     "AUC-ROC",   fmt=".3f")

    # Bounding box group
    bar(axes[1, 0], results_df["bbox_mse"].values,         "Bounding Box MSE",          "MSE",       fmt=".5f")
    bar(axes[1, 1], results_df["landmark_nme"].values,      "Landmark NME (%)",           "NME %",     fmt=".2f")
    bar(axes[1, 2], results_df["combined_loss"].values,    "Combined Loss",              "Loss ↓",    fmt=".4f")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Chart -> {output_path}")


# ─── Per-landmark breakdown plot ──────────────────────────────────────────────
def plot_per_landmark(results_df, output_path):
    """Vẽ biểu đồ MAE cho từng landmark riêng biệt."""
    if results_df.empty:
        return
    LANDMARK_NAMES = ["L.Eye", "R.Eye", "Nose", "L.Mouth", "R.Mouth"]

    n_models = len(results_df)
    x = np.arange(5)
    width = 0.8 / n_models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (_, row) in enumerate(results_df.iterrows()):
        maes = row["per_landmark_mae"]
        short = row["model_name"].replace("face_detect_model_vps_finetune_", "").replace(".pth", "")
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, maes, width * 0.9, label=short, color=colors[i], edgecolor="black", linewidth=0.4)

    ax.set_xlabel("Landmark")
    ax.set_ylabel("MAE (normalized)")
    ax.set_title("Per-Landmark MAE Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(LANDMARK_NAMES)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Per-landmark chart -> {output_path}")


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate all model checkpoints")
    parser.add_argument("--checkpoint_dir",  default="models/checkpoints",
                        help="Thư mục chứa checkpoint .pth")
    parser.add_argument("--root_checkpoint_dir", default="models",
                        help="Thư mục gốc chứa checkpoints (ưu tiên hơn checkpoint_dir)")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Số sample tối đa dùng để đánh giá (None = toàn bộ)")
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir",  default="results", help="Thư mục lưu kết quả")
    parser.add_argument("--output_csv",  default=None,      help="File CSV (mặc định: results/evaluation_results.csv)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = args.output_csv or os.path.join(args.output_dir, "evaluation_results.csv")

    # Tìm checkpoint dir ưu tiên
    for d in [args.root_checkpoint_dir, args.checkpoint_dir, "models", "."]:
        ckpts = find_checkpoints(d)
        if ckpts:
            ckpt_dir = d
            break
    else:
        print("[ERROR] Không tìm thấy checkpoint nào!")
        sys.exit(1)

    print(f"[Checkpoints] Directory: {ckpt_dir}")
    checkpoints = find_checkpoints(ckpt_dir)
    print(f"[Checkpoints] Found {len(checkpoints)}:")
    for c in checkpoints:
        print(f"  - {c}")

    # Chuẩn bị dataset
    print(f"[Dataset] Loading validation set from {LABEL_CSV} ...")
    try:
        val_dataset = CelebAValDataset(LABEL_CSV, IMG_DIR, partition=1, neg_ratio=0.3)
    except Exception as e:
        print(f"[WARN] Partition 1 failed ({e}), thử partition=0...")
        val_dataset = CelebAValDataset(LABEL_CSV, IMG_DIR, partition=0, neg_ratio=0.3)

    if args.max_samples:
        # Giữ nguyên phân bố: lấy mẫu stride từ dataset gốc
        val_dataset.data = val_dataset.data.iloc[:args.max_samples].reset_index(drop=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    print(f"[Dataset] {len(val_dataset)} validation samples, batch_size={args.batch_size}")

    # Đánh giá từng model
    rows = []
    for ckpt_path in checkpoints:
        model_name = os.path.basename(ckpt_path)
        print(f"\n{'='*60}")
        print(f"[{model_name}]")
        print(f"{'='*60}")

        model = FaceDetectMultiTask().to(device)
        model = load_model_weights(model, ckpt_path)
        model = model.to(device)

        metrics = evaluate_model(model, val_loader, device)

        # In summary
        print(f"  Classification : acc={metrics['cls_accuracy']:.4f}  "
              f"F1={metrics['cls_f1']:.4f}  AUC={metrics['cls_auc_roc']:.4f}")
        print(f"  Bounding Box   : MSE={metrics['bbox_mse']:.5f}  MAE={metrics['bbox_mae']:.5f}")
        print(f"  Landmark       : MSE={metrics['landmark_mse']:.5f}  "
              f"MAE={metrics['landmark_mae']:.5f}  NME={metrics['landmark_nme']:.2f}%")
        print(f"  Combined Loss  : {metrics['combined_loss']:.4f}")

        # Save per-landmark MAE as comma-separated string
        per_lm = ",".join(f"{v:.6f}" for v in metrics.pop("per_landmark_mae"))
        row = {"model_name": model_name, "checkpoint_path": ckpt_path, **metrics, "per_landmark_mae": per_lm}
        rows.append(row)

        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc = __import__("gc")
        gc.collect()

    # Build DataFrame
    results_df = pd.DataFrame(rows)
    cols_order = [
        "model_name", "checkpoint_path",
        "n_samples", "n_positive", "n_negative",
        "cls_accuracy", "cls_precision", "cls_recall", "cls_f1",
        "cls_auc_roc", "cls_avg_precision", "cls_loss",
        "bbox_mse", "bbox_mae", "bbox_loss",
        "landmark_mse", "landmark_mae", "landmark_nme", "landmark_loss",
        "combined_loss", "per_landmark_mae",
    ]
    results_df = results_df[[c for c in cols_order if c in results_df.columns]]
    results_df.to_csv(out_csv, index=False)
    print(f"\n[Saved] Results -> {out_csv}")

    # Plots (requires matplotlib)
    if HAS_MATPLOTLIB:
        plot_comparison(results_df, os.path.join(args.output_dir, "evaluation_metrics.png"))
        plot_per_landmark(results_df, os.path.join(args.output_dir, "evaluation_per_landmark.png"))
    else:
        print("[SKIP] Charts not generated (matplotlib not installed)")

    # Rank models
    print(f"\n{'='*60}")
    print("RANKING (sorted by Combined Loss ↓)")
    print(f"{'='*60}")
    ranked = results_df.sort_values("combined_loss").reset_index(drop=True)
    print(ranked[["model_name", "combined_loss", "cls_f1", "landmark_nme"]].to_string(index=False))


if __name__ == "__main__":
    main()
