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


def augment_hflip(image: np.ndarray, landmarks: np.ndarray):
    """Horizontal flip image (H,W,C) + mirror landmarks in-place."""
    image = image[:, ::-1, :].copy()          # flip along width axis
    landmarks = landmarks.copy()
    landmarks[0::2] = 1.0 - landmarks[0::2]  # mirror all x coords
    for a, b in _FLIP_PAIRS:                  # swap left↔right pairs
        landmarks[a], landmarks[b] = landmarks[b], landmarks[a]
    return image, landmarks


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
print(f"Đang sử dụng thiết bị: {device}")

BATCH_SIZE = 32#4GB VRAM
LEARNING_RATE = 0.001
EPOCHS = 10
IMAGE_SIZE = 224 # Kích thước chuẩn cho MobileNet

IMG_DIR   = os.path.join("celebA_dataset", "img_align_celeba", "img_align_celeba")
LABEL_CSV = "labels.csv"

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

        self.data    = df
        self.img_dir = img_dir
        self.augment = augment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]

        # --- Load & resize ---
        img_path = os.path.join(self.img_dir, row['image_id'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image.astype(np.float32) / 255.0  # shape (H,W,C) float32

        # --- Landmarks: normalize to [0,1] ---
        landmark_raw = row[['lefteye_x',  'lefteye_y',
                             'righteye_x', 'righteye_y',
                             'nose_x',     'nose_y',
                             'leftmouth_x','leftmouth_y',
                             'rightmouth_x','rightmouth_y']].values.astype(np.float32)
        landmarks = landmark_raw.copy()
        landmarks[0::2] /= IMG_W  # x coords
        landmarks[1::2] /= IMG_H  # y coords

        # --- Augmentation (train only) ---
        if self.augment:
            if random.random() < 0.5:
                image, landmarks = augment_hflip(image, landmarks)
            image = augment_color_jitter(image)

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        class_label = np.array([1.0], dtype=np.float32)
        return (
            torch.tensor(image),
            (torch.tensor(class_label), torch.tensor(landmarks))
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
    # Khởi tạo mô hình và đẩy lên GPU
    model = FaceDetectMultiTask().to(device)
    
    # Hàm mất mát và Bộ tối ưu hóa
    criterion_class = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_dataset = CelebADataset(LABEL_CSV, IMG_DIR, partition=0, augment=True)
    val_dataset   = CelebADataset(LABEL_CSV, IMG_DIR, partition=1, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train: {len(train_dataset)} (augment=True) | Val: {len(val_dataset)}")
    
    print("Bắt đầu huấn luyện...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for images, (class_labels, landmarks) in pbar:

            images       = images.to(device)
            class_labels = class_labels.to(device)
            landmarks    = landmarks.to(device)

            optimizer.zero_grad()
            class_out, _, landmark_out = model(images)

            loss_class    = criterion_class(class_out, class_labels)
            loss_landmark = criterion_reg(landmark_out, landmarks)
            loss_total    = loss_class + loss_landmark

            loss_total.backward()
            optimizer.step()
            total_loss += loss_total.item()

            # Update progress bar với loss hiện tại
            pbar.set_postfix(loss=f"{loss_total.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, (class_labels, landmarks) in val_loader:
                images       = images.to(device)
                class_labels = class_labels.to(device)
                landmarks    = landmarks.to(device)

                class_out, _, landmark_out = model(images)

                loss_class    = criterion_class(class_out, class_labels)
                loss_landmark = criterion_reg(landmark_out, landmarks)
                val_loss     += (loss_class + loss_landmark).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train: {avg_loss:.6f} | Val: {avg_val_loss:.6f}")


    # Lưu trí khôn
    torch.save(model.state_dict(), 'face_detect_model_withval2.pth')
    print("Đã lưu mô hình thành công vào file 'face_detect_model_withval.pth2'!")

if __name__ == '__main__':
    train_model()