import cv2
import numpy as np
import pandas as pd
import os
import random
from train import augment_rotate_and_scale, IMG_DIR, LABEL_CSV

def debug_3d_augmentation(num_samples=1, variations=4):
    df = pd.read_csv(LABEL_CSV)
    # Lấy các mẫu positive (có mặt)
    df_pos = df[df['partition'] == 0].head(10) 
    
    for i in range(num_samples):
        row = df_pos.iloc[i]
        img_path = os.path.join(IMG_DIR, row['image_id'])
        image = cv2.imread(img_path)
        if image is None: continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Lấy landmarks gốc (đang ở pixel)
        lm_raw = row[['lefteye_x',  'lefteye_y',
                      'righteye_x', 'righteye_y',
                      'nose_x',     'nose_y',
                      'leftmouth_x','leftmouth_y',
                      'rightmouth_x','rightmouth_y']].values.astype(np.float32)
        
        h, w = image.shape[:2]
        # Normalize landmarks để dùng với hàm augment_rotate_and_scale
        landmarks = np.zeros(10, dtype=np.float32)
        landmarks[0::2] = lm_raw[0::2] / w
        landmarks[1::2] = lm_raw[1::2] / h
        
        # BBox giả định (không quan trọng lắm cho việc xem landmark)
        bbox = np.array([0, 0, 1, 1], dtype=np.float32)

        # Lưu ảnh gốc để so sánh
        orig_view = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for j in range(0, 10, 2):
            cv2.circle(orig_view, (int(landmarks[j]*w), int(landmarks[j+1]*h)), 3, (0, 255, 0), -1)
        cv2.imwrite(f"aug_step_0_original.jpg", orig_view)
        print("Saved original image: aug_step_0_original.jpg")

        # Tạo các bản tilt khác nhau
        for v in range(1, variations + 1):
            # Áp dụng tilt (xoay max 20 độ, scale 0.8-1.2)
            aug_img, aug_lm, _ = augment_rotate_and_scale(image.copy(), landmarks.copy(), bbox.copy())
            
            # Vẽ landmark đã được transform
            aug_view = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            h_a, w_a = aug_view.shape[:2]
            for j in range(0, 10, 2):
                # Tọa độ sau khi tilt
                x = int(aug_lm[j] * w_a)
                y = int(aug_lm[j+1] * h_a)
                cv2.circle(aug_view, (x, y), 3, (0, 0, 255), -1) # Màu đỏ là sau khi tilt
            
            output_name = f"aug_step_{v}_tilt.jpg"
            cv2.imwrite(output_name, aug_view)
            print(f"Saved augmented variation {v}: {output_name}")

if __name__ == "__main__":
    # Tạo folder demo nếu cần hoặc lưu trực tiếp tại root
    debug_3d_augmentation()
