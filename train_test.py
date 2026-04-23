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

# ==========================================
# 1. THIẾT LẬP MÔI TRƯỜNG & THÔNG SỐ
# ==========================================
# Tự động chọn GPU nếu có, nếu không thì dùng CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

BATCH_SIZE = 16  # Phù hợp cho 4GB VRAM
LEARNING_RATE = 0.001
EPOCHS = 10
IMAGE_SIZE = 224 # Kích thước chuẩn cho MobileNet

# ==========================================
# 2. CHUẨN BỊ DỮ LIỆU (DATASET)
# ==========================================
class CelebADataset(Dataset):
    def __init__(self, csv_file, img_dir):
        # Giả sử bạn đã dùng Pandas gộp file txt thành file labels.csv
        self.data_frame = pd.read_csv(csv_file) 
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Đọc tên file ảnh và nhãn từ file CSV
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        image = cv2.imread(img_name)
        
        # Đổi kích thước ảnh về 224x224 và chuẩn hóa (chia 255)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image.astype(np.float32) / 255.0
        
        # PyTorch yêu cầu định dạng (Kênh màu, Chiều cao, Chiều rộng)
        image = np.transpose(image, (2, 0, 1))
        
        # Lấy nhãn (giả sử cột 1 là class, 2-5 là bbox, 6-15 là 5 điểm)
        class_label = np.array([self.data_frame.iloc[idx, 1]], dtype=np.float32)
        bbox = self.data_frame.iloc[idx, 2:6].values.astype(np.float32)
        landmarks = self.data_frame.iloc[idx, 6:16].values.astype(np.float32)

        return torch.tensor(image), (torch.tensor(class_label), torch.tensor(bbox), torch.tensor(landmarks))

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
    
    # Khởi tạo DataLoader (Bạn cần thay đường dẫn thực tế trên máy mình)
    # dataset = CelebADataset(csv_file='labels.csv', img_dir='img_align_celeba/')
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("Bắt đầu huấn luyện...")
    for epoch in range(EPOCHS):
        # Dòng này mô phỏng vòng lặp dataloader (khi bạn chạy thật hãy bỏ comment ở trên)
        # for images, (class_labels, bboxes, landmarks) in dataloader:
            
            # --- MÔ PHỎNG DỮ LIỆU ĐỂ BẠN CHẠY THỬ CODE KHÔNG LỖI ---
            images = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
            class_labels = torch.randint(0, 2, (BATCH_SIZE, 1)).float().to(device)
            bboxes = torch.rand(BATCH_SIZE, 4).to(device)
            landmarks = torch.rand(BATCH_SIZE, 10).to(device)
            # -----------------------------------------------------

            # 1. Làm sạch đạo hàm
            optimizer.zero_grad()
            
            # 2. Lan truyền xuôi (Forward)
            pred_class, pred_bbox, pred_landmark = model(images)
            
            # 3. Tính Loss tổng hợp
            loss_class = criterion_class(pred_class, class_labels)
            loss_bbox = criterion_reg(pred_bbox, bboxes)
            loss_landmark = criterion_reg(pred_landmark, landmarks)
            
            # Bạn có thể nhân thêm hệ số nếu muốn ưu tiên nhánh nào hơn
            loss_total = loss_class + loss_bbox + loss_landmark 
            
            # 4. Lan truyền ngược (Backward)
            loss_total.backward()
            
            # 5. Cập nhật trọng số
            optimizer.step()
            
            # In ra màn hình quá trình học
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Tổng Loss: {loss_total.item():.4f}")

    # Lưu trí khôn
    torch.save(model.state_dict(), 'face_detect_model.pth')
    print("Đã lưu mô hình thành công vào file 'face_detect_model.pth'!")

if __name__ == '__main__':
    train_model()