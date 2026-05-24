import torch
import cv2
import numpy as np
import os
import random
from train import FaceDetectMultiTask, IMAGE_SIZE, IMG_DIR

def test_inference(model_path, num_images=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FaceDetectMultiTask()
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    all_images = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    selected_images = random.sample(all_images, num_images)

    for img_name in selected_images:
        img_path = os.path.join(IMG_DIR, img_name)
        img_orig = cv2.imread(img_path)
        if img_orig is None: continue

        h_orig, w_orig = img_orig.shape[:2]
        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
        tensor = torch.tensor(img_input.transpose(2,0,1)).unsqueeze(0).to(device)

        with torch.no_grad():
            class_out, _, landmark_out = model(tensor)

        score = torch.sigmoid(class_out).item()
        lm = landmark_out.cpu().numpy()[0]

        print(f"Image: {img_name} | Score: {score:.4f}")
        
        # Chỉ vẽ nếu score > 0.5
        if score > 0.5:
            # Denormalize về pixel ảnh gốc
            for i in range(0, 10, 2):
                x = int(lm[i] * w_orig)
                y = int(lm[i+1] * h_orig)
                cv2.circle(img_orig, (x, y), 3, (0, 0, 255), -1)
            
            # Lưu ảnh kết quả thay vì cv2.imshow nếu không có UI
            output_name = f"result_{img_name}"
            cv2.imwrite(output_name, img_orig)
            print(f"Saved result to {output_name}")

if __name__ == "__main__":
    test_inference('face_detect_model_withval8.pth')
