"""
test_recognize_webcam.py — Face Detection + ArcFace Recognition via Webcam

Pipeline:
  1. Detect face + 5-point landmarks (FaceDetectMultiTask)
  2. Align face to 112x112 (AffinePartial2D to STD_POINTS)
  3. Extract ArcFace embedding (FaceRecognizeNet, 128D)
  4. Match against registered embeddings via cosine distance

Controls:
  'r'       - Register current face as Person_N
  '+' / '-' - Increase / decrease threshold in real-time
  'c'       - Clear all registered faces
  'q'       - Quit
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os
from torchvision import transforms
from PIL import Image

from train_v8 import FaceDetectMultiTask, IMAGE_SIZE as DET_SIZE
from train_recognize import FaceRecognizeNet

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DETECT_MODEL_PATH = os.path.join("models", "checkpoints", "face_detect_model_vps_finetune_v9.pth")
RECOG_MODEL_PATH  = os.path.join("models", "checkpoints", "face_recognize_arcface.pth")
EMBEDDING_SIZE    = 128

# Standard 5-point coords for 112x112 aligned face (InsightFace convention)
STD_POINTS = np.float32([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
])

DETECT_THRESH   = 0.85
# Start with a more lenient threshold. Use '+'/'-' in runtime to calibrate.
# Typical same-person cosine distance: 0.02 ~ 0.15
# Typical diff-person cosine distance: 0.15 ~ 0.50
RECOG_THRESH    = 0.30
THRESH_STEP     = 0.002

CROP_SCALE = 2.14
CROP_AR    = 1.22
LM_ALPHA   = 0.4
GRACE_FRAMES = 8

# Transform for recognition model (same as training)
RECOG_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ─────────────────────────────────────────────
# Model Loaders
# ─────────────────────────────────────────────
def load_detect_model(path, device):
    model = FaceDetectMultiTask()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    return model.eval().to(device)

def load_recog_model(path, device):
    model = FaceRecognizeNet(embedding_size=EMBEDDING_SIZE)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    return model.eval().to(device)

# ─────────────────────────────────────────────
# Detection helpers (two-pass same as webcam_test_v2)
# ─────────────────────────────────────────────
def detect_coarse(model, frame, device):
    fh, fw = frame.shape[:2]
    regions, coords = [], []
    for scale in [1.0, 0.8, 0.6, 0.4]:
        sz = int(fh * scale)
        step = max(1, int(sz * 0.4))
        for ry in range(0, fh - sz + 1, step):
            for rx in range(0, fw - sz + 1, step):
                coords.append((rx, ry, sz))
                crop = frame[ry:ry+sz, rx:rx+sz]
                rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                img  = cv2.resize(rgb, (DET_SIZE, DET_SIZE)).astype(np.float32) / 255.0
                regions.append(img.transpose(2, 0, 1))
    if not regions:
        return None
    t = torch.from_numpy(np.array(regions, dtype=np.float32)).to(device)
    with torch.no_grad():
        cls, bbox, _ = model(t)
    scores = torch.sigmoid(cls).cpu().numpy().squeeze(-1)
    if scores.ndim == 0:
        scores = np.array([scores])
    bi = int(np.argmax(scores))
    bs = float(scores[bi])
    if bs > DETECT_THRESH:
        rx, ry, sz = coords[bi]
        b = bbox.cpu().numpy()[bi]
        return (rx + b[0]*sz, ry + b[1]*sz, b[2]*sz, b[3]*sz, bs)
    return None

def detect_refine(model, frame, fx, fy, fw, fh, device):
    fH, fW = frame.shape[:2]
    cw = max(10, int(fw * CROP_SCALE))
    ch = max(10, int(cw * CROP_AR))
    cx = int(fx + fw/2 - cw/2)
    cy = int(fy + fh*0.4 - ch*0.51)

    pl, pt = max(0, -cx), max(0, -cy)
    pr = max(0, (cx+cw) - fW)
    pb = max(0, (cy+ch) - fH)
    if pl or pt or pr or pb:
        frame = cv2.copyMakeBorder(frame, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
        cx += pl; cy += pt
    crop = frame[cy:cy+ch, cx:cx+cw]

    rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img  = cv2.resize(rgb, (DET_SIZE, DET_SIZE)).astype(np.float32) / 255.0
    t    = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        cls, _, lm = model(t)
    score = torch.sigmoid(cls).item()
    lm_np = np.clip(lm.cpu().numpy()[0], 0.0, 1.0)
    return score, lm_np, cx - pl, cy - pt, cw, ch  # restore original coords

# ─────────────────────────────────────────────
# Face Alignment
# ─────────────────────────────────────────────
def align_face(frame, lm_abs):
    """Warp face to canonical 112x112 using 5-point similarity transform."""
    src_pts = np.float32([[lm_abs[i*2], lm_abs[i*2+1]] for i in range(5)])
    tform, _ = cv2.estimateAffinePartial2D(src_pts, STD_POINTS, method=cv2.LMEDS)
    if tform is None:
        return None
    return cv2.warpAffine(frame, tform, (112, 112), borderValue=0)

# ─────────────────────────────────────────────
# Get embedding
# ─────────────────────────────────────────────
@torch.no_grad()
def get_embedding(model, aligned_bgr, device):
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    t   = RECOG_TRANSFORM(img).unsqueeze(0).to(device)
    return model.get_embedding(t)  # L2-normalized, shape [1, 128]

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading detection model...  ", end="", flush=True)
    det_model = load_detect_model(DETECT_MODEL_PATH, device)
    print("OK")

    print("Loading recognition model...", end="", flush=True)
    rec_model = load_recog_model(RECOG_MODEL_PATH, device)
    print("OK")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam!"); return

    registered = {}   # name -> embedding tensor [1,128]
    face_ctr   = [1]  # mutable counter
    thresh     = [RECOG_THRESH]

    cached_det    = None
    ema_lm        = None
    no_det_count  = 0
    register_next = [False]

    print("\nControls:")
    print("  r       — Register current face")
    print("  + / -   — Increase / decrease threshold (current: {:.2f})".format(thresh[0]))
    print("  c       — Clear all registrations")
    print("  q       — Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # ── Detection ────────────────────────────────────────
        if cached_det is None:
            det = detect_coarse(det_model, frame, device)
            if det:
                cached_det = det
                no_det_count = 0
            else:
                ema_lm = None
                cv2.putText(display, "Scanning for face...", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)

        if cached_det is not None:
            fx, fy, fw, fh, _ = cached_det
            score, lm_norm, cx, cy, cw, ch = detect_refine(det_model, frame, fx, fy, fw, fh, device)

            if score < DETECT_THRESH * 0.8:
                no_det_count += 1
                if no_det_count > GRACE_FRAMES:
                    cached_det = None; ema_lm = None
            else:
                no_det_count = 0
                # Convert norm lm → absolute pixel coords
                lm_abs = np.zeros(10, np.float32)
                for i in range(5):
                    lm_abs[i*2]   = lm_norm[i*2]   * cw + cx
                    lm_abs[i*2+1] = lm_norm[i*2+1] * ch + cy


                if ema_lm is None:
                    ema_lm = lm_abs
                else:
                    ema_lm = LM_ALPHA * lm_abs + (1-LM_ALPHA) * ema_lm

                # Update tracking box
                px, py = ema_lm[0::2], ema_lm[1::2]
                cw2 = (max(px)-min(px)) * 2.5
                ch2 = (max(py)-min(py)) * 3.0
                mx, my = sum(px)/5, sum(py)/5
                if cached_det is None:
                    cached_det = (mx-cw2/2, my-ch2/2, cw2, ch2, score)
                else:
                    _, _, ow, oh, _ = cached_det
                    sw = 0.85*ow + 0.15*cw2
                    sh = 0.85*oh + 0.15*ch2
                    cached_det = (mx-sw/2, my-sh/2, sw, sh, score)

                # Draw face box
                bx1 = int(min(px) - (max(px)-min(px))*0.6)
                bx2 = int(max(px) + (max(px)-min(px))*0.6)
                by1 = int(min(py) - (max(py)-min(py))*1.0)
                by2 = int(max(py) + (max(py)-min(py))*0.7)

                # Draw 5 landmarks
                for i in range(5):
                    lx, ly = int(ema_lm[i*2]), int(ema_lm[i*2+1])
                    cv2.circle(display, (lx, ly), 4, (0, 220, 255), -1)

                # ── Recognition ──────────────────────────────
                aligned = align_face(frame, ema_lm)
                if aligned is not None:
                    emb = get_embedding(rec_model, aligned, device)

                    # Register
                    if register_next[0]:
                        name = f"Person_{face_ctr[0]}"
                        # Store average of 5 snapshots for robustness
                        registered[name] = emb.clone()
                        face_ctr[0] += 1
                        register_next[0] = False
                        print(f"  --> Registered: {name}")

                    # Match
                    identity   = "Unknown"
                    box_color  = (0, 0, 200)
                    best_dist  = 9999.0
                    all_dists  = {}

                    for name, reg_emb in registered.items():
                        d = 1.0 - F.cosine_similarity(emb, reg_emb).item()
                        all_dists[name] = d
                        if d < best_dist:
                            best_dist = d

                    if registered and best_dist < thresh[0]:
                        identity  = min(all_dists, key=all_dists.get)
                        box_color = (0, 200, 0)

                    # Draw bounding box
                    cv2.rectangle(display, (bx1, by1), (bx2, by2), box_color, 2)

                    # Main label
                    label = f"{identity}  dist={best_dist:.3f}"
                    cv2.putText(display, label, (bx1, by1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)

                    # DEBUG: Show distances to all registered faces (helps calibrate threshold)
                    for i, (n, d) in enumerate(all_dists.items()):
                        color = (0,200,0) if d < thresh[0] else (100,100,100)
                        cv2.putText(display, f"  {n}: {d:.4f}", (10, 200+i*22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

                    # Preview aligned face (top-right)
                    preview = cv2.resize(aligned, (80, 80))
                    h_disp  = display.shape[0]
                    display[10:90, -90:-10] = preview
                    cv2.rectangle(display, (display.shape[1]-90, 10),
                                  (display.shape[1]-10, 90), (200,200,200), 1)

        # ── HUD ──────────────────────────────────────────────
        n_reg = len(registered)
        hud = (f"Registered: {n_reg}  |  Threshold: {thresh[0]:.3f}  |  "
               f"r=Register  +/-=Threshold  c=Clear  q=Quit")
        cv2.putText(display, hud, (6, display.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 100), 1)

        cv2.imshow("ArcFace Recognition (Webcam)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            register_next[0] = True
        elif key == ord('+') or key == ord('='):
            thresh[0] = min(1.0, round(thresh[0] + THRESH_STEP, 3))
            print(f"  Threshold -> {thresh[0]:.3f}")
        elif key == ord('-'):
            thresh[0] = max(0.01, round(thresh[0] - THRESH_STEP, 3))
            print(f"  Threshold -> {thresh[0]:.3f}")
        elif key == ord('c'):
            registered.clear()
            face_ctr[0] = 1
            print("  Cleared all registrations.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
