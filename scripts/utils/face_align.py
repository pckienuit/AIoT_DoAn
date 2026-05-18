import cv2
import numpy as np
import torch

# Tọa độ 5 điểm chuẩn (Reference Points) 112x112 — chuẩn ArcFace/InsightFace
# Thứ tự: Mắt trái, Mắt phải, Mũi, Mép miệng trái, Mép miệng phải
REFERENCE_FACIAL_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


def align_face(img: np.ndarray, landmarks: np.ndarray,
               output_size: tuple = (112, 112)) -> np.ndarray:
    """
    Cắt và căn chỉnh khuôn mặt dựa trên 5 điểm mốc (landmarks).

    Args:
        img:        numpy array ảnh gốc (H, W, 3) — RGB
        landmarks:  numpy array shape (5, 2) — tọa độ pixel (x, y) của 5 điểm:
                    [lefteye, righteye, nose, leftmouth, rightmouth]
        output_size: kích thước ảnh đầu ra (default 112x112 cho ArcFace)

    Returns:
        aligned_face: numpy array (output_size[1], output_size[0], 3) đã căn chỉnh
    """
    # Ước lượng Affine Transform khớp 5 điểm thực tế → 5 điểm chuẩn
    tform, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32),
        REFERENCE_FACIAL_POINTS,
        method=cv2.LMEDS
    )

    if tform is None:
        # Fallback: chỉ crop BBox nếu không ước lượng được
        return cv2.resize(img, output_size)

    aligned = cv2.warpAffine(
        img, tform, output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return aligned


def extract_aligned_face_from_v9(model_v9, image_tensor: torch.Tensor,
                                  raw_image: np.ndarray,
                                  conf_threshold: float = 0.5) -> tuple:
    """
    Pipeline đầy đủ: Ảnh gốc -> Mô hình v9 -> Cắt & Căn chỉnh khuôn mặt.

    Args:
        model_v9:       Mô hình FaceDetectMultiTaskV9 đã load weights
        image_tensor:   Tensor (1, 3, 224, 224) — ảnh đầu vào cho v9
        raw_image:      numpy array (H, W, 3) — ảnh gốc chưa resize để crop
        conf_threshold: Ngưỡng xác suất tìm thấy khuôn mặt

    Returns:
        (aligned_face, message)
        aligned_face: numpy array 112x112 hoặc None nếu không tìm thấy mặt
        message:      chuỗi mô tả kết quả
    """
    model_v9.eval()
    with torch.no_grad():
        class_out, bbox_out, landmark_out = model_v9(image_tensor)

    score = torch.sigmoid(class_out[0]).item()
    if score < conf_threshold:
        return None, f"Không tìm thấy khuôn mặt (score={score:.3f})"

    h, w = raw_image.shape[:2]
    # Landmark: (1, 10) -> (5, 2), chuẩn hóa 0-1 -> pixel
    lm_norm = landmark_out[0].cpu().numpy().reshape(5, 2)
    landmarks_pixel = lm_norm * np.array([w, h], dtype=np.float32)

    aligned = align_face(raw_image, landmarks_pixel)
    return aligned, f"OK (score={score:.3f})"
