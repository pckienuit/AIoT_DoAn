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

# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------
# Landmark vector layout (normalized [0,1]):
#   [0,1]  lefteye   [2,3]  righteye  [4,5]  nose
#   [6,7]  leftmouth [8,9]  rightmouth
_FLIP_PAIRS = [(0, 2), (1, 3), (6, 8), (7, 9)]  # index pairs to swap on H-flip


def augment_hflip(image: np.ndarray, landmarks: np.ndarray, bbox: np.ndarray):
    """Horizontal flip image (H,W,C) + mirror landmarks and bbox in-place."""
    image = image[:, ::-1, :].copy()          # flip along width axis
    landmarks = landmarks.copy()
    landmarks[0::2] = 1.0 - landmarks[0::2]  # mirror all x coords
    for a, b in _FLIP_PAIRS:                  # swap left↔right pairs
        landmarks[a], landmarks[b] = landmarks[b], landmarks[a]
    bbox = bbox.copy()
    bbox[0] = 1.0 - (bbox[0] + bbox[2])
    return image, landmarks, bbox


def augment_color_jitter(image: np.ndarray,
                         brightness: float = 0.3,
                         contrast: float = 0.3,
                         saturation: float = 0.2) -> np.ndarray:
    """Random color jitter on float32 image [0,1] in RGB."""
    image = image.copy()
    # Brightness
    b = 1.0 + random.uniform(-brightness, brightness)
    image = np.clip(image * b, 0.0, 1.0)
    # Contrast
    c = 1.0 + random.uniform(-contrast, contrast)
    mean = image.mean()
    image = np.clip((image - mean) * c + mean, 0.0, 1.0)
    # Saturation (convert to HSV-like by scaling per channel)
    s = 1.0 + random.uniform(-saturation, saturation)
    gray = image.mean(axis=2, keepdims=True)
    image = np.clip(gray + s * (image - gray), 0.0, 1.0)
    return image.astype(np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 32           # 4GB VRAM
LEARNING_RATE = 0.001     # 0.001 for scratch; 0.0003 for fine-tuning
EPOCHS = 50
EARLY_STOP_PATIENCE = 12  # more patience for fine-tuning
LM_LOSS_WEIGHT = 10.0     # landmark regression priority (5->10)
IMAGE_SIZE = 224

# Auto-detect: Kaggle (/kaggle/input) hoặc local
def _find_kaggle_paths(root: str):
    """
    Scan đệ quy trong /kaggle/input để tìm:
      - labels.csv
      - thư mục img_align_celeba chứa ảnh .jpg
    Trả về (label_csv_path, img_dir_path) hoặc raise nếu không tìm thấy.
    """
    label_csv = img_dir = None
    for dirpath, dirnames, filenames in os.walk(root):
        if label_csv is None and "labels.csv" in filenames:
            label_csv = os.path.join(dirpath, "labels.csv")
        if img_dir is None and os.path.basename(dirpath) == "img_align_celeba":
            img_dir = dirpath
        if label_csv and img_dir:
            break

    if label_csv is None:
        raise FileNotFoundError(
            f"Khong tim thay labels.csv trong {root}\n"
            f"Cau truc hien tai:\n" +
            "\n".join(f"  {dp}" for dp, _, _ in os.walk(root))
        )
    if img_dir is None:
        raise FileNotFoundError(
            f"Khong tim thay thu muc img_align_celeba trong {root}"
        )
    # CelebA gốc có cấu trúc 2 cấp: img_align_celeba/img_align_celeba/
    # Nếu bên trong còn một img_align_celeba nữa thì dùng cấp trong
    nested = os.path.join(img_dir, "img_align_celeba")
    if os.path.isdir(nested):
        img_dir = nested
    return label_csv, img_dir


_KAGGLE_INPUT = "/kaggle/input"
if os.path.exists(_KAGGLE_INPUT):
    LABEL_CSV, IMG_DIR = _find_kaggle_paths(_KAGGLE_INPUT)
    MODEL_OUT   = "/kaggle/working/face_detect_model_withval7.pth"
    # Resume from checkpoint if available; None = train from scratch.
    RESUME_FROM = None  # withval5 deleted — training from scratch
    print(f"[Kaggle] LABEL_CSV : {LABEL_CSV}")
    print(f"[Kaggle] IMG_DIR   : {IMG_DIR}")
else:
    IMG_DIR     = os.path.join("celebA_dataset", "img_align_celeba", "img_align_celeba")
    LABEL_CSV   = "labels.csv"
    MODEL_OUT   = "face_detect_model_withval7.pth"
    RESUME_FROM = "face_detect_model_withval5.pth"

IMG_W, IMG_H = 178, 218

class CelebADataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, partition: int, augment: bool = False):
        """
        Args:
            partition: 0=train, 1=val, 2=test
            augment:   True -> apply random H-flip + color jitter (train only)
        """
        df = pd.read_csv(csv_file)
        df = df[df['partition'] == partition].reset_index(drop=True)

        self.data      = df
        self.img_dir   = img_dir
        self.augment   = augment
        # For val/test: negative sampling is deterministic per index.
        # For train:    True random (augment=True implies stochastic).
        self.deterministic_neg = not augment

    def __len__(self) -> int:
        return len(self.data)

    def _generate_negative_crop(
        self,
        image_full: np.ndarray,
        bbox_raw: np.ndarray,
        rng: random.Random | None = None,
    ) -> np.ndarray:
        """Random background crop with <10% face overlap.

        Args:
            rng: If provided (val/test), use this seeded RNG so the crop is
                 identical across epochs -> stable val_loss.
                 If None (train), use the global random module.
        """
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
                   bw: float, bh: float):
        """Simulate inference pipeline: expand face BBox by 2.14x with zero-pad."""
        CROP_SCALE = 2.14
        CROP_AR    = 1.22  # 218/178 CelebA aspect ratio

        crop_w = max(10, int(bw * CROP_SCALE))
        crop_h = max(10, int(crop_w * CROP_AR))
        crop_x = int(bx + bw / 2 - crop_w / 2)
        crop_y = int(by + bh * 0.4 - crop_h * 0.51)  # eye alignment

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

        # --- Load & resize ---
        img_path = os.path.join(self.img_dir, row['image_id'])
        image_full = cv2.imread(img_path)
        image_full = cv2.cvtColor(image_full, cv2.COLOR_BGR2RGB)
        
        bbox_raw = row[['x_1', 'y_1', 'width', 'height']].values.astype(np.float32)

        # 20% chance of negative sample.
        # Val/test: fully deterministic per index (same is_negative AND same crop
        # geometry every epoch) -> truly stable val_loss.
        if self.deterministic_neg:
            rng = random.Random(idx)           # fixed seed covers ALL randomness below
            is_negative = (rng.random() < 0.2)
        else:
            rng = None
            is_negative = (random.random() < 0.2)

        if is_negative:
            # Pass rng so crop coordinates are also deterministic for val/test.
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

        # --- Positive: simulate inference crop pipeline ---
        bx, by, bw, bh = bbox_raw
        face_crop, crop_x, crop_y, crop_w, crop_h = self._make_crop(image_full, bx, by, bw, bh)
        image = cv2.resize(face_crop, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0

        # BBox normalized within the crop (same coordinate system as inference)
        bbox = np.array([
            np.clip((bx - crop_x) / crop_w, 0.0, 1.0),
            np.clip((by - crop_y) / crop_h, 0.0, 1.0),
            np.clip(bw / crop_w, 0.0, 1.0),
            np.clip(bh / crop_h, 0.0, 1.0),
        ], dtype=np.float32)

        # Landmarks normalized within the crop
        landmark_raw = row[['lefteye_x',  'lefteye_y',
                             'righteye_x', 'righteye_y',
                             'nose_x',     'nose_y',
                             'leftmouth_x','leftmouth_y',
                             'rightmouth_x','rightmouth_y']].values.astype(np.float32)
        landmarks = np.zeros(10, dtype=np.float32)
        landmarks[0::2] = np.clip((landmark_raw[0::2] - crop_x) / crop_w, 0.0, 1.0)
        landmarks[1::2] = np.clip((landmark_raw[1::2] - crop_y) / crop_h, 0.0, 1.0)

        # --- Augmentation (train only) ---
        if self.augment:
            if random.random() < 0.5:
                image, landmarks, bbox = augment_hflip(image, landmarks, bbox)
            image = augment_color_jitter(image)

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        class_label = np.array([1.0], dtype=np.float32)
        return (
            torch.tensor(image),
            (torch.tensor(class_label), torch.tensor(bbox), torch.tensor(landmarks))
        )

# ==========================================
# 3. KIẾN TRÚC MÔ HÌNH (MODEL)
# ==========================================
class FaceDetectMultiTask(nn.Module):
    def __init__(self):
        super(FaceDetectMultiTask, self).__init__()
        # Dùng MobileNetV2 làm xương sống (nhẹ, nhanh, rất tốt cho máy tính cá nhân)
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.backbone = mobilenet.features # Trích xuất phần lõi
        
        # Kích thước đầu ra của MobileNetV2 sau khi qua lớp Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3 nhánh đầu ra
        self.class_head = nn.Linear(1280, 1)    # Có mặt hay không
        self.bbox_head = nn.Linear(1280, 4)     # 4 tọa độ hộp
        self.landmark_head = nn.Linear(1280, 10) # 10 tọa độ (5 điểm x 2)

    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features)
        features = torch.flatten(features, 1) # Làm phẳng ma trận thành vector 1280 chiều
        
        class_out = self.class_head(features)
        bbox_out = self.bbox_head(features)
        landmark_out = self.landmark_head(features)
        
        return class_out, bbox_out, landmark_out

# ==========================================
# 4. KHỞI TẠO & HUẤN LUYỆN
# ==========================================
def train_model():
    model = FaceDetectMultiTask().to(device)

    # Resume from checkpoint (fine-tune) if configured.
    if RESUME_FROM and os.path.isfile(RESUME_FROM):
        state = torch.load(RESUME_FROM, map_location=device)
        model.load_state_dict(state)
        print(f"[Resume] Loaded weights from '{RESUME_FROM}'")
    else:
        print("[Train] Starting from scratch (ImageNet pretrained backbone).")

    # Multi-GPU: DataParallel for Kaggle Dual T4.
    if torch.cuda.device_count() > 1:
        print(f"[Multi-GPU] Su dung {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Hàm mất mát và Bộ tối ưu hóa
    criterion_class = nn.BCEWithLogitsLoss()
    criterion_reg = nn.SmoothL1Loss() # SmoothL1Loss ổn định hơn MSE cho regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # CosineAnnealingLR: smooth LR decay, avoids the aggressive StepLR drops.
    # eta_min keeps a small residual LR so training doesn't stall completely.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
    
    train_dataset = CelebADataset(LABEL_CSV, IMG_DIR, partition=0, augment=True)
    val_dataset   = CelebADataset(LABEL_CSV, IMG_DIR, partition=1, augment=False)
    # Tăng num_workers để tải data song song với GPU đang train
    n_workers = min(4, os.cpu_count() or 0)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=n_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=n_workers, pin_memory=True)
    print(f"Train: {len(train_dataset)} (augment=True) | Val: {len(val_dataset)}")
    
    print("Bat dau huan luyen...")
    best_val_loss  = float('inf')
    patience_count = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for images, (class_labels, bboxes, landmarks) in pbar:

            images       = images.to(device)
            class_labels = class_labels.to(device)
            bboxes       = bboxes.to(device)
            landmarks    = landmarks.to(device)

            optimizer.zero_grad()
            class_out, bbox_out, landmark_out = model(images)

            loss_class    = criterion_class(class_out, class_labels)
            
            # Chỉ tính loss bbox và landmark cho các sample có mặt người (class_labels == 1)
            mask = (class_labels == 1.0).squeeze(-1)
            if mask.sum() > 0:
                loss_bbox     = criterion_reg(bbox_out[mask], bboxes[mask])
                loss_landmark = criterion_reg(landmark_out[mask], landmarks[mask])
            else:
                loss_bbox     = torch.tensor(0.0, device=device)
                loss_landmark = torch.tensor(0.0, device=device)
                
            # Weighted loss: landmark weight raised to LM_LOSS_WEIGHT for better NME.
            loss_total = loss_class * 1.0 + loss_bbox * 5.0 + loss_landmark * LM_LOSS_WEIGHT

            loss_total.backward()
            optimizer.step()
            total_loss += loss_total.item()

            # Update progress bar với chi tiết từng loại loss
            pbar.set_postfix(
                cls=f"{loss_class.item():.4f}",
                box=f"{loss_bbox.item():.4f}",
                lm=f"{loss_landmark.item():.4f}"
            )

        scheduler.step() # Cập nhật Learning Rate
        avg_loss = total_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, (class_labels, bboxes, landmarks) in val_loader:
                images       = images.to(device)
                class_labels = class_labels.to(device)
                bboxes       = bboxes.to(device)
                landmarks    = landmarks.to(device)

                class_out, bbox_out, landmark_out = model(images)

                loss_class    = criterion_class(class_out, class_labels)
                
                mask = (class_labels == 1.0).squeeze(-1)
                if mask.sum() > 0:
                    loss_bbox     = criterion_reg(bbox_out[mask], bboxes[mask])
                    loss_landmark = criterion_reg(landmark_out[mask], landmarks[mask])
                else:
                    loss_bbox     = torch.tensor(0.0, device=device)
                    loss_landmark = torch.tensor(0.0, device=device)
                    
                # Same weights as train for fair comparison.
                val_loss += (loss_class * 1.0 + loss_bbox * 5.0 + loss_landmark * LM_LOSS_WEIGHT).item()

        avg_val_loss = val_loss / len(val_loader)
        current_lr   = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train: {avg_loss:.8f} | Val: {avg_val_loss:.8f} | LR: {current_lr:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss  = avg_val_loss
            patience_count = 0
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, MODEL_OUT)
            print(f"--> Saved best model | val_loss: {best_val_loss:.8f}")
        else:
            patience_count += 1
            print(f"    No improvement ({patience_count}/{EARLY_STOP_PATIENCE})")
            if patience_count >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    print(f"Training done. Best model saved at '{MODEL_OUT}'")

if __name__ == '__main__':
    train_model()