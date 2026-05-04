"""
webcam_test_v2.py — Model-only Two-Pass Pipeline (GPU)

Pass 1 (Coarse): Run model on multi-scale frame crops to detect face → raw bbox
Pass 2 (Refine): Aligned crop (2.14x, same as training) → precise landmarks
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import os
import time
from train_v8 import FaceDetectMultiTask, IMAGE_SIZE

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_PATH    = os.path.join("models", "checkpoints", "face_detect_model_vps_finetune_v9.pth")
LANDMARK_NAMES = ["L.Eye", "R.Eye", "Nose", "L.Mouth", "R.Mouth"]

COLOR_DET   = (80,  255, 80)    # Green: coarse detection box
COLOR_CROP  = (200, 200, 0)     # Cyan: aligned crop sent to model
COLOR_LM    = (0,   220, 255)   # Yellow: landmarks
COLOR_TEXT  = (255, 255, 255)   # White
COLOR_ERR   = (0,   0,   220)   # Red: no face

# EMA smoothing (0 = frozen, 1 = no smoothing)
LM_ALPHA   = 0.40   # smooth the landmarks

# Detection
DETECT_THRESH = 0.85  # increased to filter out bright ceiling lights
CROP_SCALE    = 2.14  # must match training (_make_crop)
CROP_AR       = 1.22  # must match training (218/178)

# Anti-jitter
GRACE_FRAMES = 8    # keep last detection for N frames before "no face"


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
def load_model(path: str, device: torch.device) -> FaceDetectMultiTask:
    model = FaceDetectMultiTask()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint

    # Backward compat: old linear landmark head
    if "landmark_head.weight" in state_dict:
        print("[Auto-Fix] Legacy weights detected — reverting landmark head to nn.Linear")
        model.landmark_head = nn.Linear(1280, 10)

    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
def _to_tensor(bgr_img: np.ndarray, device: torch.device) -> torch.Tensor:
    """BGR ndarray → normalised GPU tensor [1, 3, H, W]."""
    rgb     = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
    tensor  = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(device)


# ─────────────────────────────────────────────
# Pass 1 — Coarse detection
# ─────────────────────────────────────────────
def coarse_detect(model: FaceDetectMultiTask, frame: np.ndarray, device: torch.device):
    """
    Batched sliding-window grid scan to find faces anywhere in the frame.
    Returns (face_x, face_y, face_w, face_h, score) in frame coords, or None.
    """
    fh, fw = frame.shape[:2]
    
    regions = []
    coords  = []
    
    # Scale: size of square crop relative to frame height
    scales = [1.0, 0.8, 0.6, 0.4]
    
    for scale in scales:
        sz = int(fh * scale)
        step = max(1, int(sz * 0.4))  # 60% overlap
        
        # Collect crops
        for ry in range(0, fh - sz + 1, step):
            for rx in range(0, fw - sz + 1, step):
                # Ensure we also cover the exact right/bottom edges if the step doesn't divide evenly
                coords.append((rx, ry, sz))
                region = frame[ry:ry + sz, rx:rx + sz]
                
                rgb     = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
                regions.append(resized.transpose(2, 0, 1))

    if not regions:
        return None
        
    # Batch inference
    tensor = torch.from_numpy(np.array(regions, dtype=np.float32)).to(device)
    
    with torch.no_grad():
        cls_out, bbox_out, _ = model(tensor)
        
    scores = torch.sigmoid(cls_out).cpu().numpy().squeeze(-1)
    
    if len(scores.shape) == 0:
        scores = np.array([scores])
        
    best_idx   = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    
    # The model might trigger false positives on bright lights (like the ceiling).
    # Setting threshold high to filter out background noise since grid search has many windows.
    if best_score > DETECT_THRESH:
        rx, ry, sz = coords[best_idx]
        b = bbox_out.cpu().numpy()[best_idx]
        
        best_det = (
            rx + b[0] * sz,
            ry + b[1] * sz,
            b[2] * sz,
            b[3] * sz,
            best_score,
        )
        return best_det

    return None  # None if no region exceeded threshold


# ─────────────────────────────────────────────
# Pass 2 — Aligned crop + refinement
# ─────────────────────────────────────────────
def refine_predict(model: FaceDetectMultiTask, frame: np.ndarray,
                   face_x: float, face_y: float, face_w: float, face_h: float,
                   device: torch.device):
    """
    Build aligned crop identical to training (_make_crop with 2.14x / 1.22 AR).
    Returns (cls_score, bbox[4], lm[10], crop_x, crop_y, crop_w, crop_h).
    """
    fh, fw = frame.shape[:2]

    crop_w = max(10, int(face_w * CROP_SCALE))
    crop_h = max(10, int(crop_w * CROP_AR))
    crop_x = int(face_x + face_w / 2 - crop_w / 2)
    crop_y = int(face_y + face_h * 0.4 - crop_h * 0.51)

    # Use BORDER_REPLICATE to avoid sharp black edges (matches training exactly)
    pad_left   = max(0, -crop_x)
    pad_top    = max(0, -crop_y)
    pad_right  = max(0, (crop_x + crop_w) - fw)
    pad_bottom = max(0, (crop_y + crop_h) - fh)

    if pad_left or pad_top or pad_right or pad_bottom:
        padded_frame = cv2.copyMakeBorder(
            frame,
            top=pad_top, bottom=pad_bottom,
            left=pad_left, right=pad_right,
            borderType=cv2.BORDER_REPLICATE
        )
        # Shift crop coordinates since the frame grew
        c_x = crop_x + pad_left
        c_y = crop_y + pad_top
        face_crop = padded_frame[c_y:c_y+crop_h, c_x:c_x+crop_w]
    else:
        face_crop = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

    tensor = _to_tensor(face_crop, device)

    with torch.no_grad():
        cls_out, bbox_out, lm_out = model(tensor)

    cls_score = torch.sigmoid(cls_out).item()
    bbox      = np.clip(bbox_out.cpu().numpy()[0], 0.0, 1.0)
    lm        = np.clip(lm_out.cpu().numpy()[0],   0.0, 1.0)

    return cls_score, bbox, lm, crop_x, crop_y, crop_w, crop_h


# ─────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────
def draw_landmarks(frame, lm):
    for i, name in enumerate(LANDMARK_NAMES):
        px = int(lm[i * 2])
        py = int(lm[i * 2 + 1])
        cv2.circle(frame, (px, py), 5, COLOR_LM, -1)
        cv2.putText(frame, name, (px + 6, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, COLOR_TEXT, 1)


def draw_det_box(frame, fx, fy, fw, fh, score):
    """Draw smoothed coarse detection box (pass-1 result)."""
    x1, y1 = int(fx), int(fy)
    x2, y2 = int(fx + fw), int(fy + fh)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_DET, 2)
    cv2.putText(frame, f"face: {score:.2f}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_DET, 1)


def draw_crop_outline(frame, crop_x, crop_y, crop_w, crop_h):
    """Draw the aligned crop region sent to pass-2 (thin cyan box)."""
    cv2.rectangle(frame,
                  (crop_x, crop_y),
                  (crop_x + crop_w, crop_y + crop_h),
                  COLOR_CROP, 1)


def draw_refined_bbox(frame, lm):
    """Derive a stable, full-face bounding box from the facial landmarks."""
    pts_x = [lm[i*2] for i in range(5)]
    pts_y = [lm[i*2+1] for i in range(5)]
    
    # Core feature box (eyes to mouth)
    min_x, max_x = min(pts_x), max(pts_x)
    min_y, max_y = min(pts_y), max(pts_y)
    
    core_w = max_x - min_x
    core_h = max_y - min_y
    
    # Expand to cover full head
    x1 = int(min_x - core_w * 0.5)
    x2 = int(max_x + core_w * 0.5)
    y1 = int(min_y - core_h * 0.8) # up to forehead
    y2 = int(max_y + core_h * 0.6) # down to chin
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_DET, 2)
    cv2.putText(frame, "face", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_DET, 1)


def draw_hud(frame, score: float, fps: float, device_name: str):
    h = frame.shape[0]
    cv2.putText(frame, f"face: {score:.2f}  |  FPS: {fps:.1f}  |  {device_name}",
                (6, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
    cv2.putText(frame, "Model-only Pipeline  (Q to quit)",
                (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_TEXT, 1)


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
def run_webcam(model: FaceDetectMultiTask, device: torch.device):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam!")
        return

    print("Model-only two-pass pipeline — press Q to quit")
    print("  Pass 1: multi-scale coarse detect  |  Pass 2: aligned-crop refinement")

    # EMA states (pass-2 based — same domain as training)
    ema_lm:   np.ndarray | None = None

    # Anti-jitter state
    frame_idx    = 0
    cached_det   = None   # last successful coarse detection
    no_det_count = 0      # frames since last successful detection
    last_score   = 0.0

    t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ── Pass 1: Coarse detection (only if lost) ──────────
        if cached_det is None:
            det = coarse_detect(model, frame, device)
            if det is not None:
                cached_det   = det
                no_det_count = 0
            else:
                # No face detected
                cv2.putText(frame, "Scanning for face...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.80, COLOR_ERR, 2)
                ema_lm   = None
                last_score = 0.0
                
                # Draw HUD and skip to next frame
                t_now = time.perf_counter()
                fps   = 1.0 / max(1e-6, t_now - t_prev)
                t_prev = t_now
                draw_hud(frame, last_score, fps, device.type.upper())
                cv2.imshow("Model-only: Detect + Landmark  (Q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

        # ── Pass 2: Active Tracking ──────────────────────────
        face_x, face_y, face_w, face_h, _ = cached_det

        # Use the tracking box to build the aligned crop
        p2_score, bbox, lm, crop_x, crop_y, crop_w, crop_h = \
            refine_predict(model, frame, face_x, face_y, face_w, face_h, device)

        if p2_score < DETECT_THRESH * 0.8:  # slightly lower threshold to maintain tracking
            no_det_count += 1
            if no_det_count > GRACE_FRAMES:
                cached_det = None
                ema_lm     = None
            # Keep drawing old EMA if within grace frames
        else:
            # CRITICAL FIX: Anchor tracking box size to actual facial landmarks!
            # Using model bounding box in a feedback loop causes exponential divergence.
            raw_lm_abs = np.zeros(10, dtype=np.float32)
            for i in range(5):
                raw_lm_abs[i*2]   = lm[i*2] * crop_w + crop_x
                raw_lm_abs[i*2+1] = lm[i*2+1] * crop_h + crop_y

            # 1. Reject impossible landmarks (e.g. false positive ceiling light)
            pts_x = [raw_lm_abs[i*2] for i in range(5)]
            pts_y = [raw_lm_abs[i*2+1] for i in range(5)]
            core_w = max(pts_x) - min(pts_x)
            core_h = max(pts_y) - min(pts_y)
            
            # If features collapse into a tiny point, it's not a face
            if core_w < 15 or core_h < 15 or core_w > frame.shape[1] * 0.8:
                no_det_count = GRACE_FRAMES + 1 # Force immediate reset
                cached_det = None
                ema_lm = None
                continue
                
            no_det_count = 0

            # 2. EMA on absolute frame coordinates
            if ema_lm is None:
                ema_lm = raw_lm_abs
            else:
                ema_lm = LM_ALPHA * raw_lm_abs + (1 - LM_ALPHA) * ema_lm

            # 3. Update the ACTIVE TRACKER for the next frame using smoothed landmarks
            e_pts_x = [ema_lm[i*2] for i in range(5)]
            e_pts_y = [ema_lm[i*2+1] for i in range(5)]
            e_core_w = max(e_pts_x) - min(e_pts_x)
            e_core_h = max(e_pts_y) - min(e_pts_y)
            e_center_x = sum(e_pts_x) / 5.0
            e_center_y = sum(e_pts_y) / 5.0
            
            # CelebA box is roughly 2.2x the eye-to-mouth core distance
            new_w = e_core_w * 2.2
            new_h = e_core_h * 2.2
            
            if cached_det is None:
                cached_det = (e_center_x - new_w/2, e_center_y - new_h/2, new_w, new_h, p2_score)
            else:
                _, _, old_w, old_h, _ = cached_det
                # Heavy EMA on scale (size) to prevent "breathing" / divergence
                smooth_w = 0.9 * old_w + 0.1 * new_w
                smooth_h = 0.9 * old_h + 0.1 * new_h
                smooth_x = e_center_x - smooth_w / 2
                smooth_y = e_center_y - smooth_h / 2
                cached_det = (smooth_x, smooth_y, smooth_w, smooth_h, p2_score)

        if ema_lm is not None:
            # Draw pass-2 bbox (derived from absolute landmarks) and crop outline
            draw_refined_bbox(frame, ema_lm)
            draw_crop_outline(frame, crop_x, crop_y, crop_w, crop_h)
            draw_landmarks(frame, ema_lm)
            last_score = p2_score

        # ── HUD ──────────────────────────────────────────────
        t_now = time.perf_counter()
        fps   = 1.0 / max(1e-6, t_now - t_prev)
        t_prev = t_now
        draw_hud(frame, last_score, fps, device.type.upper())

        cv2.imshow("Model-only: Detect + Landmark  (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = load_model(MODEL_PATH, device)
    print("Model loaded. Starting webcam...")

    run_webcam(model, device)
