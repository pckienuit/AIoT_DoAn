import os
import cv2
import numpy as np
import pandas as pd
import random

def create_calibration_dataset(csv_file, img_dir, output_dir, num_samples=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Dang doc file labels.csv...")
    df = pd.read_csv(csv_file)
    # Lấy ra các ảnh trong tập train (partition == 0)
    df_train = df[df['partition'] == 0].reset_index(drop=True)
    
    # Lấy mẫu ngẫu nhiên (chọn seed cố định để dễ debug nghiệm)
    rng = random.Random(42)
    indices = rng.sample(range(len(df_train)), num_samples)
    
    print(f"Bat dau trich xuat {num_samples} anh cho Calibration Dataset...")
    
    count = 0
    for idx in indices:
        row = df_train.iloc[idx]
        img_path = os.path.join(img_dir, row['image_id'])
        
        if not os.path.exists(img_path):
            continue
            
        # Đọc ảnh (OpenCV đọc mặc định BGR)
        image_full = cv2.imread(img_path)
        
        # Ảnh trong tập img_align_celeba đã được cắt cúp và căn chỉnh sẵn (178x218)
        # Không cần dùng bbox từ CSV (bbox đó dành cho ảnh chưa align)
        
        # Resize trực tiếp về đúng kích thước đầu vào của mô hình (224x224)
        IMAGE_SIZE = 224
        face_crop_resized = cv2.resize(image_full, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Lưu file
        out_name = f"calib_{count:03d}_{row['image_id']}"
        out_path = os.path.join(output_dir, out_name)
        
        # Lưu định dạng JPG
        cv2.imwrite(out_path, face_crop_resized)
        count += 1
        
        if count % 20 == 0:
            print(f"  Da xuat {count}/{num_samples} anh...")
            
    print(f"Hoan thanh! File duoc luu tai thu muc: {output_dir}")
    print(f"Trong MaixVision, hay nho cau hinh: Mean = [0,0,0], Scale = [0.00392, 0.00392, 0.00392] (tuc la 1/255) vi model cua ban normalize theo khoang [0, 1].")

if __name__ == "__main__":
    csv_file = "labels.csv"
    img_dir = os.path.join("celebA_dataset", "img_align_celeba", "img_align_celeba")
    output_dir = "calibration_data"
    
    create_calibration_dataset(csv_file, img_dir, output_dir, num_samples=100)
