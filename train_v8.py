import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import cv2
import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm
import gc

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# ---------------------------------------------------------------------------
# Augmentation helpers — v8: LIGHT augmentation, NO rotation/scale
# Rotation/scale transform distorts landmark coordinates and degrades NME.
# ---------------------------------------------------------------------------
_FLIP_PAIRS = [(0, 2), (1, 3), (6, 8), (7, 9)]


def augment_hflip(image: np.ndarray, landmarks: np.ndarray, bbox: np.ndarray):
    image = image[:, ::-1, :].copy()
    landmarks = landmarks.copy()
    landmarks[0::2] = 1.0 - landmarks[0::2]
    for a, b in _FLIP_PAIRS:
        landmarks[a], landmarks[b] = landmarks[b], landmarks[a]
    bbox = bbox.copy()
    bbox[0] = 1.0 - (bbox[0] + bbox[2])
    return image, landmarks, bbox


def augment_color_jitter(image: np.ndarray,
                         brightness: float = 0.15,
                         contrast: float = 0.15,
                         saturation: float = 0.1) -> np.ndarray:
    """Light color jitter on float32 image [0,1] in RGB."""
    image = image.copy()
    b = 1.0 + random.uniform(-brightness, brightness)
    image = np.clip(image * b, 0.0, 1.0)
    c = 1.0 + random.uniform(-contrast, contrast)
    mean = image.mean()
    image = np.clip((image - mean) * c + mean, 0.0, 1.0)
    s = 1.0 + random.uniform(-saturation, saturation)
    gray = image.mean(axis=2, keepdims=True)
    image = np.clip(gray + s * (image - gray), 0.0, 1.0)
    return image.astype(np.float32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def wing_loss(pred: torch.Tensor, target: torch.Tensor,
              w: float = 10.0, eps: float = 2.0) -> torch.Tensor:
    """
    Wing Loss (Feng et al., CVPR 2018).
    Scale lỗi lên pixel (224) để log curve hoạt động đúng với tọa độ [0,1].
    """
    SCALE = 224.0
    x = (pred * SCALE - target * SCALE).abs()
    C = w - w * math.log(1.0 + w / eps)
    loss = torch.where(x < w,
                       w * torch.log(1.0 + x / eps),
                       x - C)
    return loss.mean() / SCALE


class FocalLandmarkLoss(nn.Module):
    """Focal loss giup model tap trung vao hard samples (IOD nho, occlusion)."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, scale: float = 224.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.scale = scale

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).abs() * self.scale
        pt = torch.exp(-diff)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * diff
        return loss.mean() / self.scale


BATCH_SIZE = 64
LEARNING_RATE = 1e-5
EPOCHS = 60
EARLY_STOP_PATIENCE = 20
LM_WING_WEIGHT  = 0.7
LM_FOCAL_WEIGHT = 1.0
LM_LOSS_WEIGHT  = 20.0
GRAD_CLIP_NORM  = 1.0
WARMUP_EPOCHS    = 5
IMAGE_SIZE       = 224

# Progressive Unfreeze Schedule
UNFREEZE_PHASE2 = 10
UNFREEZE_PHASE3 = 20
BACKBONE_LR_RATIO = 0.1


def _find_dataset_paths(root: str):
    label_csv = img_dir = None
    for dirpath, dirnames, filenames in os.walk(root):
        if label_csv is None and "labels.csv" in filenames:
            label_csv = os.path.join(dirpath, "labels.csv")
        if img_dir is None and os.path.basename(dirpath) == "img_align_celeba":
            img_dir = dirpath
        if label_csv and img_dir:
            break
    if img_dir:
        nested = os.path.join(img_dir, "img_align_celeba")
        if os.path.isdir(nested):
            img_dir = nested
    return label_csv, img_dir


_KAGGLE_INPUT = "/kaggle/input"
if os.path.exists(_KAGGLE_INPUT):
    LABEL_CSV, IMG_DIR = _find_dataset_paths(_KAGGLE_INPUT)
    MODEL_OUT   = "/kaggle/working/face_detect_model_vps_finetune_v8.pth"
    RESUME_FROM = "/kaggle/input/face-detect-weights/face_detect_model_vps_finetune_v3.pth"
    print(f"[Kaggle] LABEL_CSV : {LABEL_CSV}")
    print(f"[Kaggle] IMG_DIR   : {IMG_DIR}")
else:
    LABEL_CSV, IMG_DIR = _find_dataset_paths(".")
    if not LABEL_CSV or not IMG_DIR:
        print("[Warning] Khong tim thay dataset tu dong.")
        IMG_DIR   = os.path.join("celebA_dataset", "img_align_celeba", "img_align_celeba")
        LABEL_CSV = "labels.csv"

    MODEL_OUT   = os.path.join("models", "checkpoints", "face_detect_model_vps_finetune_v8.pth")
    RESUME_FROM = os.path.join("models", "checkpoints", "face_detect_model_vps_finetune_v3.pth")

IMG_W, IMG_H = 178, 218


class CelebADataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, partition: int, augment: bool = False):
        df = pd.read_csv(csv_file)
        df = df[df['partition'] == partition].reset_index(drop=True)
        self.data      = df
        self.img_dir   = img_dir
        self.augment   = augment
        self.deterministic_neg = not augment

    def __len__(self) -> int:
        return len(self.data)

    def _generate_negative_crop(
        self,
        image_full: np.ndarray,
        bbox_raw: np.ndarray,
        rng: random.Random | None = None,
    ) -> np.ndarray:
        _ri = rng.randint if rng else random.randint
        H, W, _ = image_full.shape
        bx, by, bw, bh = int(bbox_raw[0]), int(bbox_raw[1]), int(bbox_raw[2]), int(bbox_raw[3])

        for _ in range(10):
            crop_size = _ri(40, max(40, min(W, H) // 2))
            cx = _ri(0, max(0, W - crop_size))
            cy = _ri(0, max(0, H - crop_size))

            inter_x1 = max(cx, bx)
            inter_y1 = max(cy, by)
            inter_x2 = min(cx + crop_size, bx + bw)
            inter_y2 = min(cy + crop_size, by + bh)

            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                if inter_area < (crop_size * crop_size * 0.1):
                    return image_full[cy:cy+crop_size, cx:cx+crop_size]
            else:
                return image_full[cy:cy+crop_size, cx:cx+crop_size]

        return image_full[0:30, 0:30]

    def _make_crop(self, image_full: np.ndarray, bx: float, by: float,
                   bw: float, bh: float, augment_shift: bool = False):
        CROP_SCALE = 2.14
        CROP_AR    = 1.22

        crop_w = max(10, int(bw * CROP_SCALE))
        crop_h = max(10, int(crop_w * CROP_AR))
        crop_x = int(bx + bw / 2 - crop_w / 2)
        crop_y = int(by + bh * 0.4 - crop_h * 0.51)

        if augment_shift:
            crop_x += int(random.uniform(-0.03, 0.03) * crop_w)
            crop_y += int(random.uniform(-0.03, 0.03) * crop_h)

        H, W = image_full.shape[:2]
        face_crop = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

        src_x1, src_y1 = max(0, crop_x),          max(0, crop_y)
        src_x2, src_y2 = min(W, crop_x + crop_w), min(H, crop_y + crop_h)
        dst_x1 = max(0, -crop_x)
        dst_y1 = max(0, -crop_y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if src_x2 > src_x1 and src_y2 > src_y1:
            face_crop[dst_y1:dst_y2, dst_x1:dst_x2] = image_full[src_y1:src_y2, src_x1:src_x2]

        return face_crop, crop_x, crop_y, crop_w, crop_h

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.img_dir, row['image_id'])
        image_full = cv2.imread(img_path)
        image_full = cv2.cvtColor(image_full, cv2.COLOR_BGR2RGB)
        bbox_raw = row[['x_1', 'y_1', 'width', 'height']].values.astype(np.float32)

        if self.deterministic_neg:
            rng = random.Random(idx)
            is_negative = (rng.random() < 0.2)
        else:
            rng = None
            is_negative = (random.random() < 0.2)

        if is_negative:
            crop = self._generate_negative_crop(image_full, bbox_raw, rng=rng)
            image = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
            image = image.astype(np.float32) / 255.0

            if self.augment:
                image = augment_color_jitter(image)
                if random.random() < 0.5:
                    image = image[:, ::-1, :].copy()

            image = np.transpose(image, (2, 0, 1))
            class_label = np.array([0.0], dtype=np.float32)
            bbox = np.zeros(4, dtype=np.float32)
            landmarks = np.zeros(10, dtype=np.float32)
            return (
                torch.tensor(image),
                (torch.tensor(class_label), torch.tensor(bbox), torch.tensor(landmarks))
            )

        bx, by, bw, bh = bbox_raw
        face_crop, crop_x, crop_y, crop_w, crop_h = self._make_crop(
            image_full, bx, by, bw, bh, augment_shift=self.augment
        )
        image = cv2.resize(face_crop, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0

        bbox = np.array([
            np.clip((bx - crop_x) / crop_w, 0.0, 1.0),
            np.clip((by - crop_y) / crop_h, 0.0, 1.0),
            np.clip(bw / crop_w, 0.0, 1.0),
            np.clip(bh / crop_h, 0.0, 1.0),
        ], dtype=np.float32)

        landmark_raw = row[['lefteye_x',  'lefteye_y',
                             'righteye_x', 'righteye_y',
                             'nose_x',     'nose_y',
                             'leftmouth_x','leftmouth_y',
                             'rightmouth_x','rightmouth_y']].values.astype(np.float32)
        landmarks = np.zeros(10, dtype=np.float32)
        landmarks[0::2] = np.clip((landmark_raw[0::2] - crop_x) / crop_w, 0.0, 1.0)
        landmarks[1::2] = np.clip((landmark_raw[1::2] - crop_y) / crop_h, 0.0, 1.0)

        # --- Augmentation (v8): NHẸ NHẤT, KHÔNG rotation/scale ---
        if self.augment:
            if random.random() < 0.5:
                image, landmarks, bbox = augment_hflip(image, landmarks, bbox)
            # Light color jitter (reduced from 0.3/0.2 to 0.15/0.1)
            image = augment_color_jitter(image)

        image = np.transpose(image, (2, 0, 1))
        class_label = np.array([1.0], dtype=np.float32)
        return (
            torch.tensor(image),
            (torch.tensor(class_label), torch.tensor(bbox), torch.tensor(landmarks))
        )


# ==========================================
# MODEL — v8: MobileNetV2 (same as v3 for edge device compatibility)
# ==========================================
class FaceDetectMultiTask(nn.Module):
    def __init__(self):
        super(FaceDetectMultiTask, self).__init__()
        mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.backbone = mobilenet.features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 3 task heads
        self.class_head    = nn.Linear(1280, 1)
        self.bbox_head     = nn.Linear(1280, 4)

        # Landmark Head: MLP với Dropout (same as v3)
        self.landmark_head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features)
        features = torch.flatten(features, 1)

        class_out    = self.class_head(features)
        bbox_out     = self.bbox_head(features)
        landmark_out = self.landmark_head(features)

        return class_out, bbox_out, landmark_out


# ==========================================
# TRAINING
# ==========================================
def _set_backbone_freeze(model: nn.Module, freeze: bool):
    backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
    for p in backbone.parameters():
        p.requires_grad = not freeze


def _set_backbone_partial_freeze(model: nn.Module, unfreeze_from_block: int):
    backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
    for idx, block in enumerate(backbone):
        for p in block.parameters():
            p.requires_grad = (idx >= unfreeze_from_block)


def train_model():
    model = FaceDetectMultiTask().to(device)

    criterion_class = nn.BCEWithLogitsLoss()
    criterion_bbox  = nn.SmoothL1Loss()
    focal_landmark_loss = FocalLandmarkLoss(alpha=0.75, gamma=2.0)

    start_epoch   = 0
    best_val_loss = float('inf')
    resumed       = False

    if RESUME_FROM and os.path.isfile(RESUME_FROM):
        checkpoint = torch.load(RESUME_FROM, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        # v8 same architecture as v3 — full load
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[Resume] Da load tu '{RESUME_FROM}'.")
        start_epoch = 0
        best_val_loss = float('inf')
        resumed = True
    else:
        print("[Train] Starting from scratch (ImageNet pretrained MobileNetV2).")

    head_params = [
        {'params': model.class_head.parameters()},
        {'params': model.bbox_head.parameters()},
        {'params': model.landmark_head.parameters()},
    ]

    if resumed:
        _set_backbone_freeze(model, freeze=False)
        print("[v8] Resume Mode: Backbone FULLY UNFROZEN from start.")
        backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
        head_params.append({'params': backbone.parameters(), 'lr': LEARNING_RATE * BACKBONE_LR_RATIO})
    else:
        _set_backbone_freeze(model, freeze=True)
        print("[v8-Phase1] Backbone FROZEN — chi train Heads (epoch 1-10).")

    optimizer = optim.AdamW(head_params, lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    if torch.cuda.device_count() > 1:
        print(f"[Multi-GPU] Su dung {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    train_dataset = CelebADataset(LABEL_CSV, IMG_DIR, partition=0, augment=True)
    val_dataset   = CelebADataset(LABEL_CSV, IMG_DIR, partition=1, augment=False)

    n_workers = min(8, os.cpu_count() or 0)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=n_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=n_workers, pin_memory=True)
    print(f"Train: {len(train_dataset)} (augment=True) | Val: {len(val_dataset)}")

    print("Bat dau huan luyen v8...")
    patience_count = 0

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0

        # Progressive Unfreeze
        if not resumed:
            if epoch == UNFREEZE_PHASE2:
                _set_backbone_partial_freeze(model, unfreeze_from_block=16)
                backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
                optimizer.add_param_group({'params': list(backbone[16:].parameters()), 'lr': LEARNING_RATE * BACKBONE_LR_RATIO})
                print(f"[v8-Phase2] Mo khoa Backbone block 16-18 | LR backbone = {LEARNING_RATE * BACKBONE_LR_RATIO:.2e}")
            elif epoch == UNFREEZE_PHASE3:
                _set_backbone_freeze(model, freeze=False)
                backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
                optimizer.add_param_group({'params': list(backbone[:16].parameters()), 'lr': LEARNING_RATE * BACKBONE_LR_RATIO})
                print(f"[v8-Phase3] Mo khoa TOAN BO Backbone | LR backbone = {LEARNING_RATE * BACKBONE_LR_RATIO:.2e}")

        # Warmup
        if resumed and epoch < WARMUP_EPOCHS:
            warmup_scale = (epoch + 1) / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_scale * (LEARNING_RATE if pg is optimizer.param_groups[0] else LEARNING_RATE * BACKBONE_LR_RATIO)
            print(f"  [Warmup] Epoch {epoch+1}/{WARMUP_EPOCHS} | LR = {warmup_scale * LEARNING_RATE:.6e}")

        is_kaggle = os.path.exists(_KAGGLE_INPUT)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch", disable=is_kaggle)
        for i, (images, (class_labels, bboxes, landmarks)) in enumerate(pbar):

            images       = images.to(device)
            class_labels = class_labels.to(device)
            bboxes       = bboxes.to(device)
            landmarks    = landmarks.to(device)

            optimizer.zero_grad()
            class_out, bbox_out, landmark_out = model(images)

            loss_class = criterion_class(class_out, class_labels)

            mask = (class_labels == 1.0).squeeze(-1)
            if mask.sum() > 0:
                loss_bbox       = criterion_bbox(bbox_out[mask], bboxes[mask])
                loss_landmark_w = wing_loss(landmark_out[mask], landmarks[mask])
                loss_landmark_f = focal_landmark_loss(landmark_out[mask], landmarks[mask])
                loss_landmark   = loss_landmark_w * LM_WING_WEIGHT + loss_landmark_f * LM_FOCAL_WEIGHT
            else:
                loss_bbox     = torch.tensor(0.0, device=device)
                loss_landmark = torch.tensor(0.0, device=device)

            loss_total = loss_class * 1.0 + loss_bbox * 5.0 + loss_landmark * LM_LOSS_WEIGHT

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            total_loss += loss_total.item()

            if is_kaggle:
                if i % 50 == 0:
                    print(f"  Batch {i}/{len(train_loader)} | Loss: {loss_total.item():.4f} (cls:{loss_class.item():.4f}, box:{loss_bbox.item():.4f}, lm:{loss_landmark.item():.4f})")
            else:
                pbar.set_postfix(
                    cls=f"{loss_class.item():.4f}",
                    box=f"{loss_bbox.item():.4f}",
                    lm=f"{loss_landmark.item():.4f}"
                )

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, (class_labels, bboxes, landmarks) in val_loader:
                images       = images.to(device)
                class_labels = class_labels.to(device)
                bboxes       = bboxes.to(device)
                landmarks    = landmarks.to(device)

                class_out, bbox_out, landmark_out = model(images)

                loss_class = criterion_class(class_out, class_labels)

                mask = (class_labels == 1.0).squeeze(-1)
                if mask.sum() > 0:
                    loss_bbox       = criterion_bbox(bbox_out[mask], bboxes[mask])
                    loss_landmark_w = wing_loss(landmark_out[mask], landmarks[mask])
                    loss_landmark_f = focal_landmark_loss(landmark_out[mask], landmarks[mask])
                    loss_landmark   = loss_landmark_w * LM_WING_WEIGHT + loss_landmark_f * LM_FOCAL_WEIGHT
                else:
                    loss_bbox     = torch.tensor(0.0, device=device)
                    loss_landmark = torch.tensor(0.0, device=device)

                val_loss += (loss_class * 1.0 + loss_bbox * 5.0 + loss_landmark * LM_LOSS_WEIGHT).item()

        avg_val_loss = val_loss / len(val_loader)
        current_lr   = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train: {avg_loss:.8f} | Val: {avg_val_loss:.8f} | LR: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss  = avg_val_loss
            patience_count = 0
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
            }, MODEL_OUT)
            torch.save(model_state, MODEL_OUT.replace('.pth', '_weights.pth'))
            print(f"--> Saved best model | val_loss: {best_val_loss:.8f} | epoch: {epoch+1}")
        else:
            patience_count += 1
            print(f"    No improvement ({patience_count}/{EARLY_STOP_PATIENCE})")
            if patience_count >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

        gc.collect()

    print(f"Training done. Best model saved at '{MODEL_OUT}'")


if __name__ == '__main__':
    train_model()
