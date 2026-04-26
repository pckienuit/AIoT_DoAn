import cv2
import torch
import numpy as np
from train import FaceDetectMultiTask, IMAGE_SIZE

MODEL_PATH     = "face_detect_model_withval2.pth"
HAAR_PATH      = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

LANDMARK_NAMES = ["L.Eye", "R.Eye", "Nose", "L.Mouth", "R.Mouth"]
COLOR_BOX      = (255, 200, 0)
COLOR_LM       = (0, 255, 0)
COLOR_TEXT     = (0, 255, 255)


def load_model(path: str) -> FaceDetectMultiTask:
    model = FaceDetectMultiTask()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def predict_landmarks(model: FaceDetectMultiTask, face_crop: np.ndarray) -> np.ndarray:
    """Nhận BGR face crop → trả về landmarks shape (10,) đã denormalize về [0,1]."""
    rgb     = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
    tensor  = torch.tensor(resized.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        _, _, landmark_out = model(tensor)
    return landmark_out.numpy()[0]  # shape (10,)


def draw_landmarks_on_crop(frame: np.ndarray, lm: np.ndarray,
                            x: int, y: int, w: int, h: int) -> None:
    """Vẽ 5 landmarks lên frame gốc, trong vùng bbox (x,y,w,h)."""
    for i, name in enumerate(LANDMARK_NAMES):
        # lm[i*2], lm[i*2+1] là tọa độ [0,1] TRONG crop
        # → scale về pixel của crop → offset về frame
        px = int(lm[i * 2]     * w) + x
        py = int(lm[i * 2 + 1] * h) + y
        cv2.circle(frame, (px, py), 4, COLOR_LM, -1)
        cv2.putText(frame, name, (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_TEXT, 1)


def run_webcam(model: FaceDetectMultiTask) -> None:
    cap      = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(HAAR_PATH)

    if not cap.isOpened():
        print("Không mở được webcam!")
        return

    print("Webcam chạy — nhấn Q để thoát")
    print("Bước 1: Haar Cascade detect mặt  |  Bước 2: MobileNetV2 predict landmarks")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=8, minSize=(100, 100))

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            for (fx, fy, fw, fh) in faces:
                # 1. Tính toán kích thước crop để mô phỏng CelebA
                # CelebA: khoảng cách 2 mắt ~ 0.21 chiều rộng ảnh (38/178)
                # Haar: khoảng cách 2 mắt ~ 0.45 chiều rộng box
                # => Tỷ lệ mở rộng: 0.45 / 0.21 ≈ 2.14
                crop_w = int(fw * 2.14)
                crop_h = int(crop_w * 1.22) # Tỷ lệ 218/178 của CelebA

                # 2. Canh chỉnh vị trí (Align)
                # CelebA: mắt nằm ở 0.51 chiều cao ảnh từ trên xuống (111/218)
                # Haar: mắt thường nằm ở ~0.4 chiều cao box từ trên xuống
                crop_x = int(fx + fw / 2 - crop_w / 2)
                crop_y = int(fy + fh * 0.4 - crop_h * 0.51)

                # 3. Tạo vùng crop an toàn (có padding đen nếu tràn viền)
                # Giúp ảnh không bị méo tỷ lệ khi dí sát mặt vào viền camera
                face_crop = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
                
                src_x1 = max(0, crop_x)
                src_y1 = max(0, crop_y)
                src_x2 = min(frame.shape[1], crop_x + crop_w)
                src_y2 = min(frame.shape[0], crop_y + crop_h)

                dst_x1 = max(0, -crop_x)
                dst_y1 = max(0, -crop_y)
                dst_x2 = dst_x1 + (src_x2 - src_x1)
                dst_y2 = dst_y1 + (src_y2 - src_y1)

                if src_x2 <= src_x1 or src_y2 <= src_y1:
                    continue

                face_crop[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

                # Vẽ box để debug
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 1)
                cv2.rectangle(frame, (crop_x, crop_y), (crop_x+crop_w, crop_y+crop_h), COLOR_BOX, 2)

                # 4. Predict và vẽ landmarks
                lm = predict_landmarks(model, face_crop)
                draw_landmarks_on_crop(frame, lm, crop_x, crop_y, crop_w, crop_h)

        cv2.imshow("2-Stage Pipeline: Detect → Landmark  (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Loading model...")
    model = load_model(MODEL_PATH)
    run_webcam(model)
