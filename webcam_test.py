import cv2
import torch
import numpy as np
from train import FaceDetectMultiTask, IMAGE_SIZE

MODEL_PATH     = "face_detect_model_withval8.pth"
HAAR_PATH      = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

LANDMARK_NAMES = ["L.Eye", "R.Eye", "Nose", "L.Mouth", "R.Mouth"]
COLOR_HAAR     = (255, 0, 0)      # Blue: Haar Cascade box
COLOR_CROP     = (255, 200, 0)    # Cyan: crop region sent to model
COLOR_BBOX     = (0, 255, 0)      # Green: model predicted BBox
COLOR_LM       = (0, 220, 255)    # Yellow: landmarks
COLOR_TEXT     = (255, 255, 255)  # White: text

# EMA smoothing: 0.0 = frozen, 1.0 = no smoothing
BBOX_ALPHA = 0.25  # Low alpha -> very smooth but slightly lagged
LM_ALPHA   = 0.35


def load_model(path: str) -> FaceDetectMultiTask:
    model = FaceDetectMultiTask()
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def predict_face(model: FaceDetectMultiTask, face_crop: np.ndarray):
    """BGR face crop -> (class_score float, bbox [4], landmarks [10]) all normalized [0,1]."""
    rgb     = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
    tensor  = torch.tensor(resized.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        cls_out, bbox_out, lm_out = model(tensor)
    cls_score = torch.sigmoid(cls_out).item()
    bbox      = bbox_out.numpy()[0]   # [x, y, w, h] normalized
    landmarks = lm_out.numpy()[0]     # [10] normalized
    return cls_score, bbox, landmarks


def draw_landmarks(frame: np.ndarray, lm: np.ndarray,
                   x: int, y: int, w: int, h: int) -> None:
    """Draw 5 landmarks on the full frame, offset by crop origin (x, y)."""
    for i, name in enumerate(LANDMARK_NAMES):
        px = int(lm[i * 2]     * w) + x
        py = int(lm[i * 2 + 1] * h) + y
        cv2.circle(frame, (px, py), 5, COLOR_LM, -1)
        cv2.putText(frame, name, (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_TEXT, 1)


def draw_bbox(frame: np.ndarray, bbox: np.ndarray,
              crop_x: int, crop_y: int, crop_w: int, crop_h: int) -> None:
    """Draw model's predicted BBox (normalized to crop) onto the full frame."""
    bx = int(bbox[0] * crop_w) + crop_x
    by = int(bbox[1] * crop_h) + crop_y
    bw = int(bbox[2] * crop_w)
    bh = int(bbox[3] * crop_h)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), COLOR_BBOX, 2)


def run_webcam(model: FaceDetectMultiTask) -> None:
    cap      = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(HAAR_PATH)

    if not cap.isOpened():
        print("Cannot open webcam!")
        return

    print("Webcam running -- press Q to quit")
    print("Step 1: Haar Cascade detect face  |  Step 2: MobileNetV2 predict landmarks")

    ema_bbox: np.ndarray | None = None  # EMA state for BBox
    ema_lm:   np.ndarray | None = None  # EMA state for Landmarks

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

                # Draw boxes
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), COLOR_HAAR, 1)  # Haar
                cv2.rectangle(frame, (crop_x, crop_y), (crop_x+crop_w, crop_y+crop_h), COLOR_CROP, 1)  # Crop

                # 4. Predict & draw
                cls_score, bbox, lm = predict_face(model, face_crop)

                # EMA smoothing: reduces per-frame jitter significantly
                if ema_bbox is None:
                    ema_bbox = bbox.copy()
                    ema_lm   = lm.copy()
                else:
                    ema_bbox = BBOX_ALPHA * bbox + (1 - BBOX_ALPHA) * ema_bbox
                    ema_lm   = LM_ALPHA   * lm   + (1 - LM_ALPHA)   * ema_lm

                # Only draw Model BBox if model is confident (face detected)
                if cls_score > 0.5:
                    draw_bbox(frame, ema_bbox, crop_x, crop_y, crop_w, crop_h)

                draw_landmarks(frame, ema_lm, crop_x, crop_y, crop_w, crop_h)

                # Confidence label
                cv2.putText(frame, f"face: {cls_score:.2f}", (fx, fy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BBOX, 1)
            else:
                # Reset EMA when no face detected
                ema_bbox = None
                ema_lm   = None

        # Legend
        cv2.putText(frame, "Blue=Haar  Cyan=Crop  Green=ModelBBox", (5, frame.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("2-Stage Pipeline: Detect → Landmark  (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Loading model...")
    model = load_model(MODEL_PATH)
    run_webcam(model)
