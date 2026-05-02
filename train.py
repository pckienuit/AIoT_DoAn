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

# Tránh rò rỉ RAM (CPU) do đa luồng của OpenCV kết hợp với DataLoader
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

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


def augment_rotate_and_scale(image: np.ndarray, landmarks: np.ndarray, bbox: np.ndarray, angle_max=10, scale_range=(0.9, 1.1)):
    """
    Simulates 3D-like poses (tilt, zoom) by rotating and scaling in 2D.
    """
    h, w = image.shape[:2]
    angle = random.uniform(-angle_max, angle_max)
    scale = random.uniform(*scale_range)
    
    # Calculate rotation matrix
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Transform image
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Transform landmarks
    # landmarks shape (10,) -> points shape (5, 2)
    pts = landmarks.reshape(5, 2) * [w, h]
    # Add ones for affine transformation: [x, y, 1]
    pts_ones = np.hstack([pts, np.ones((5, 1))])
    new_pts = M.dot(pts_ones.T).T
    new_landmarks = (new_pts / [w, h]).flatten()
    
    # Transform bbox (approximate by transforming 4 corners and taking min/max)
    bx, by, bw, bh = bbox
    corners = np.array([
        [bx, by], [bx+bw, by], [bx, by+bh], [bx+bw, by+bh]
    ]) * [w, h]
    corners_ones = np.hstack([corners, np.ones((4, 1))])
    new_corners = M.dot(corners_ones.T).T
    new_corners = new_corners / [w, h]
    
    new_bx = np.min(new_corners[:, 0])
    new_by = np.min(new_corners[:, 1])
    new_bw = np.max(new_corners[:, 0]) - new_bx
    new_bh = np.max(new_corners[:, 1]) - new_by
    new_bbox = np.array([new_bx, new_by, new_bw, new_bh], dtype=np.float32)

    return image, np.clip(new_landmarks, 0, 1), np.clip(new_bbox, 0, 1)

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

# ---------------------------------------------------------------------------
# [CS2] Wing Loss — Thiết kế cho landmark detection
# Nhạy với sai số nhỏ hơn SmoothL1, ép mô hình căn chỉnh pixel cuối cùng.
# ---------------------------------------------------------------------------
def wing_loss(pred: torch.Tensor, target: torch.Tensor,
             w: float = 10.0, eps: float = 2.0) -> torch.Tensor:
    """
    Wing Loss (Feng et al., CVPR 2018).
    FIX BUG: Tọa độ ở dải [0, 1] nên x rất nhỏ (luôn < w=10), khiến Wing Loss bị thoái hóa 
    thành L1 Loss. Phải scale lỗi lên pixel (VD: 224) để log curve hoạt động đúng.
    """
    SCALE = 224.0
    x = (pred * SCALE - target * SCALE).abs()
    C = w - w * math.log(1.0 + w / eps)
    
    loss = torch.where(x < w,
                       w * torch.log(1.0 + x / eps),
                       x - C)
    return loss.mean() / SCALE

BATCH_SIZE = 64           # Giảm batch size cho VPS (RAM 15GB, CPU 20 cores)
LEARNING_RATE = 1e-5      # LR cực nhỏ để "vắt kiệt" NME
EPOCHS = 30               
EARLY_STOP_PATIENCE = 15  
LM_LOSS_WEIGHT = 15.0     # Mức 15 là đủ vì Wing Loss đã được sửa lỗi scale
GRAD_CLIP_NORM = 1.0      
WARMUP_EPOCHS = 3         
IMAGE_SIZE = 224

# Auto-detect: Kaggle (/kaggle/input) hoặc local/VPS
def _find_dataset_paths(root: str):
    """Scan đệ quy tìm labels.csv và thư mục img_align_celeba."""
    label_csv = img_dir = None
    for dirpath, dirnames, filenames in os.walk(root):
        if label_csv is None and "labels.csv" in filenames:
            label_csv = os.path.join(dirpath, "labels.csv")
        if img_dir is None and os.path.basename(dirpath) == "img_align_celeba":
            img_dir = dirpath
        if label_csv and img_dir:
            break

    # Fix lồng thư mục:
    if img_dir:
        nested = os.path.join(img_dir, "img_align_celeba")
        if os.path.isdir(nested):
            img_dir = nested
    return label_csv, img_dir

_KAGGLE_INPUT = "/kaggle/input"
if os.path.exists(_KAGGLE_INPUT):
    LABEL_CSV, IMG_DIR = _find_dataset_paths(_KAGGLE_INPUT)
    MODEL_OUT   = "/kaggle/working/face_detect_model_vps_finetune_v4.pth"
    RESUME_FROM = "/kaggle/input/face-detect-weights/face_detect_model_vps_finetune_v3.pth"
    print(f"[Kaggle] LABEL_CSV : {LABEL_CSV}")
    print(f"[Kaggle] IMG_DIR   : {IMG_DIR}")
else:
    # Trên VPS hoặc Local
    LABEL_CSV, IMG_DIR = _find_dataset_paths(".")
    if not LABEL_CSV or not IMG_DIR:
        print("[Warning] Không tìm thấy dataset tự động. Đang dùng path mặc định.")
        IMG_DIR     = os.path.join("celebA_dataset", "img_align_celeba", "img_align_celeba")
        LABEL_CSV   = "labels.csv"
    
    MODEL_OUT   = "face_detect_model_vps_finetune_v4.pth"
    RESUME_FROM = "face_detect_model_vps_finetune_v3.pth"

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
                   bw: float, bh: float, augment_shift: bool = False):
        """Simulate inference pipeline: expand face BBox by 2.14x with zero-pad."""
        CROP_SCALE = 2.14
        CROP_AR    = 1.22  # 218/178 CelebA aspect ratio

        crop_w = max(10, int(bw * CROP_SCALE))
        crop_h = max(10, int(crop_w * CROP_AR))
        crop_x = int(bx + bw / 2 - crop_w / 2)
        crop_y = int(by + bh * 0.4 - crop_h * 0.51)  # eye alignment

        if augment_shift:
            # Shift ngẫu nhiên lên đến 5% kích thước crop để giả lập translation
            crop_x += int(random.uniform(-0.05, 0.05) * crop_w)
            crop_y += int(random.uniform(-0.05, 0.05) * crop_h)

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
        face_crop, crop_x, crop_y, crop_w, crop_h = self._make_crop(image_full, bx, by, bw, bh, augment_shift=self.augment)
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
            
            # Rotation/scale mạnh hơn: 50% chance, góc max 15 độ
            if random.random() < 0.5:
                image, landmarks, bbox = augment_rotate_and_scale(image, landmarks, bbox, angle_max=15)
                
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
        self.backbone = mobilenet.features  # 19 blocks (index 0-18)

        # Kích thước đầu ra của MobileNetV2 sau khi qua lớp Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 3 nhánh đầu ra
        self.class_head    = nn.Linear(1280, 1)   # Có mặt hay không
        self.bbox_head     = nn.Linear(1280, 4)   # 4 tọa độ hộp

        # [CS1] Nâng cấp Landmark Head: Linear -> MLP với Dropout
        # Thay vì nén thẳng 1280->10 (quá đột ngột),
        # ta thêm một lớp trung gian 256 để AI "suy nghĩ" sâu hơn.
        # Dropout(0.3) ngẫu nhiên tắt 30% nơ-ron khi train -> chống overfitting.
        self.landmark_head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features)
        features = torch.flatten(features, 1)  # Làm phẳng thành vector 1280 chiều

        class_out    = self.class_head(features)
        bbox_out     = self.bbox_head(features)
        landmark_out = self.landmark_head(features)

        return class_out, bbox_out, landmark_out

# ==========================================
# 4. KHỞI TẠO & HUẤN LUYỆN
# ==========================================
# [CS3] Progressive Unfreeze Schedule
# Phase 1 (epoch 1-10):   Đóng băng toàn bộ backbone, chỉ train Heads.
# Phase 2 (epoch 11-20):  Mở khóa 3 block cuối (block 16-18).
# Phase 3 (epoch 21+):    Mở khóa toàn bộ với LR backbone = LR_HEAD / 10.
UNFREEZE_PHASE2 = 10   # epoch bắt đầu phase 2
UNFREEZE_PHASE3 = 20   # epoch bắt đầu phase 3
BACKBONE_LR_RATIO = 0.1  # backbone học chậm hơn head 10 lần


def _set_backbone_freeze(model: nn.Module, freeze: bool):
    """Đóng / mở khóa toàn bộ backbone."""
    backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
    for p in backbone.parameters():
        p.requires_grad = not freeze


def _set_backbone_partial_freeze(model: nn.Module, unfreeze_from_block: int):
    """Mở khóa các block từ `unfreeze_from_block` trở đi, đóng phần trước."""
    backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
    for idx, block in enumerate(backbone):
        for p in block.parameters():
            p.requires_grad = (idx >= unfreeze_from_block)


def train_model():
    model = FaceDetectMultiTask().to(device)

    # Hàm mất mát
    criterion_class = nn.BCEWithLogitsLoss()
    # [CS2] Wing Loss thay thế SmoothL1 cho landmark
    # (SmoothL1 vẫn dùng cho bbox vì bbox không cần độ nhạy pixel)
    criterion_bbox  = nn.SmoothL1Loss()

    # Phase 1: đóng băng backbone, chỉ train heads
    _set_backbone_freeze(model, freeze=True)
    print("[CS3-Phase1] Backbone FROZEN — chỉ train Heads.")

    # Optimizer: chia param_groups để dễ điều chỉnh LR backbone/head riêng
    head_params = [
        {'params': model.class_head.parameters()},
        {'params': model.bbox_head.parameters()},
        {'params': model.landmark_head.parameters()},
    ]
    optimizer = optim.AdamW(head_params, lr=LEARNING_RATE, weight_decay=1e-4)

    # CosineAnnealingLR: smooth LR decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    start_epoch    = 0
    best_val_loss  = float('inf')
    resumed        = False

    # Resume from checkpoint
    if RESUME_FROM and os.path.isfile(RESUME_FROM):
        checkpoint = torch.load(RESUME_FROM, map_location=device)
        
        # Determine state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Không bỏ qua landmark_head vì ta resume từ v2 đã có kiến trúc MLP
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[Resume] Đã load toàn bộ trọng số từ '{RESUME_FROM}'. Tiếp tục fine-tune.")
        
        # Bắt đầu train từ Epoch 0 vì landmark_head là hoàn toàn mới
        start_epoch = 0
        best_val_loss = float('inf')
    else:
        print("[Train] Starting from scratch (ImageNet pretrained backbone).")

    # Multi-GPU: DataParallel for Kaggle Dual T4.
    if torch.cuda.device_count() > 1:
        print(f"[Multi-GPU] Su dung {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    train_dataset = CelebADataset(LABEL_CSV, IMG_DIR, partition=0, augment=True)
    val_dataset   = CelebADataset(LABEL_CSV, IMG_DIR, partition=1, augment=False)
    
    # Tăng n_workers cho VPS (vì VPS này có 20 cores)
    n_workers = min(8, os.cpu_count() or 0)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=n_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=n_workers, pin_memory=True)
    print(f"Train: {len(train_dataset)} (augment=True) | Val: {len(val_dataset)}")
    
    print("Bat dau huan luyen...")
    patience_count = 0

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0

        # [CS3] Progressive Unfreeze: chuyển phase theo epoch
        if epoch == UNFREEZE_PHASE2:
            _set_backbone_partial_freeze(model, unfreeze_from_block=16)
            backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
            optimizer.add_param_group({'params': list(backbone[16:].parameters()), 'lr': LEARNING_RATE * BACKBONE_LR_RATIO})
            print(f"[CS3-Phase2] Mở khóa Backbone block 16-18 | LR backbone = {LEARNING_RATE * BACKBONE_LR_RATIO:.2e}")
        elif epoch == UNFREEZE_PHASE3:
            _set_backbone_freeze(model, freeze=False)
            backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
            optimizer.add_param_group({'params': list(backbone[:16].parameters()), 'lr': LEARNING_RATE * BACKBONE_LR_RATIO})
            print(f"[CS3-Phase3] Mở khóa TOÀN BỘ Backbone | LR backbone = {LEARNING_RATE * BACKBONE_LR_RATIO:.2e}")

        # Linear warmup: ramp LR from LR/10 → LR over WARMUP_EPOCHS
        # Only applies when loading weights-only (no optimizer state).
        if resumed and epoch < WARMUP_EPOCHS:
            warmup_lr = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
            print(f"  [Warmup] Epoch {epoch+1}/{WARMUP_EPOCHS} | LR = {warmup_lr:.6f}")

        is_kaggle = os.path.exists(_KAGGLE_INPUT)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch", disable=is_kaggle)
        for i, (images, (class_labels, bboxes, landmarks)) in enumerate(pbar):

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
                loss_bbox     = criterion_bbox(bbox_out[mask], bboxes[mask])           # SmoothL1 cho bbox
                loss_landmark = wing_loss(landmark_out[mask], landmarks[mask])          # [CS2] Wing Loss cho landmark
            else:
                loss_bbox     = torch.tensor(0.0, device=device)
                loss_landmark = torch.tensor(0.0, device=device)

            # Weighted loss: landmark weight raised to LM_LOSS_WEIGHT for better NME.
            loss_total = loss_class * 1.0 + loss_bbox * 5.0 + loss_landmark * LM_LOSS_WEIGHT

            loss_total.backward()
            # Clip gradients to prevent spikes when landmark weight is high.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            total_loss += loss_total.item()

            # Update progress bar với chi tiết từng loại loss
            if is_kaggle:
                if i % 50 == 0:
                    print(f"  Batch {i}/{len(train_loader)} | Loss: {loss_total.item():.4f} (cls:{loss_class.item():.4f}, box:{loss_bbox.item():.4f}, lm:{loss_landmark.item():.4f})")
            else:
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
                    loss_bbox     = criterion_bbox(bbox_out[mask], bboxes[mask])
                    loss_landmark = wing_loss(landmark_out[mask], landmarks[mask])
                else:
                    loss_bbox     = torch.tensor(0.0, device=device)
                    loss_landmark = torch.tensor(0.0, device=device)

                # Same weights as train for fair comparison.
                val_loss += (loss_class * 1.0 + loss_bbox * 5.0 + loss_landmark * LM_LOSS_WEIGHT).item()

        avg_val_loss = val_loss / len(val_loader)
        current_lr   = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train: {avg_loss:.8f} | Val: {avg_val_loss:.8f} | LR: {current_lr:.6f}")

        # Save best model — full checkpoint for proper resume later
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
            # Also save weights-only for inference/evaluate.py compatibility
            torch.save(model_state, MODEL_OUT.replace('.pth', '_weights.pth'))
            print(f"--> Saved best model | val_loss: {best_val_loss:.8f} | epoch: {epoch+1}")
        else:
            patience_count += 1
            print(f"    No improvement ({patience_count}/{EARLY_STOP_PATIENCE})")
            if patience_count >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
                
    # Dọn rác RAM CPU sau mỗi epoch
        gc.collect()

    print(f"Training done. Best model saved at '{MODEL_OUT}'")

if __name__ == '__main__':
    train_model()