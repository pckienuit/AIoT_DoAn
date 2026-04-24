# Tạo file inference_test.py
import torch
import cv2
import numpy as np
from train import FaceDetectMultiTask, IMAGE_SIZE

model = FaceDetectMultiTask()
model.load_state_dict(torch.load('face_detect_model_withval.pth', map_location='cpu'))
model.eval()

# Thử với ảnh CelebA bất kỳ
img = cv2.imread(r'celebA_dataset\img_align_celeba\img_align_celeba\000001.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
tensor = torch.tensor(img_resized.transpose(2,0,1)).unsqueeze(0)

with torch.no_grad():
    class_out, _, landmark_out = model(tensor)

print(f"Class score (sigmoid): {torch.sigmoid(class_out).item():.4f}")  # gần 1.0 = "có mặt"
print(f"Landmarks: {landmark_out.numpy()}")

# Vẽ landmarks lên ảnh gốc (178x218)
img_orig = cv2.imread(r'celebA_dataset\img_align_celeba\img_align_celeba\000001.jpg')
lm = landmark_out.numpy()[0]

# Denormalize về pixel
IMG_W, IMG_H = 178, 218
points = [(int(lm[i]*IMG_W), int(lm[i+1]*IMG_H)) for i in range(0, 10, 2)]

for (x, y) in points:
    cv2.circle(img_orig, (x, y), 3, (0, 0, 255), -1)

cv2.imshow("Landmarks", img_orig)
cv2.waitKey(0)

