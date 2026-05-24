from maix import camera, display, image, nn, app, sys, tensor
import gc
import json
import math
import os
import time

from config import load_config
from sync_cache import CacheManager
from display import (
    draw_match_result, draw_status, draw_hud, draw_no_match
)

# =====================================================================
# CONFIG & CACHE INIT
# =====================================================================
CFG = load_config()
cache_mgr = CacheManager(CFG)

# Active flight — first in list by default. Can be changed at runtime
# by writing a new value to /root/active_flight.txt
ACTIVE_FLIGHT_FILE = "/root/active_flight.txt"


def read_active_flight() -> int:
    if os.path.exists(ACTIVE_FLIGHT_FILE):
        try:
            with open(ACTIVE_FLIGHT_FILE, "r") as f:
                return int(f.read().strip())
        except Exception:
            pass
    ids = CFG.get("flight_ids", [1])
    return ids[0] if ids else 1


# =====================================================================
# MODEL PATHS
# =====================================================================
MODEL_DIR = "/root/models"
FACE_DB_PATH = "/root/face_db.json"
REGISTER_REQUEST_PATH = "/root/register_name.txt"
CLEAR_DB_FLAG_PATH = "/root/clear_face_db.flag"

if sys.device_name().lower() == "maixcam2":
    FACE_DET = nn.YOLO11(model=MODEL_DIR + "/yolo11s_face.mud", dual_buff=False)
else:
    FACE_DET = nn.YOLOv8(model=MODEL_DIR + "/yolov8n_face.mud", dual_buff=False)

LM_MODEL    = nn.NN(MODEL_DIR + "/face_detect_v9.mud")
RECOG_MODEL = nn.NN(MODEL_DIR + "/face_recognize_arcface_p3.mud")

OUT_CLASS_ALIASES    = ["class_out_Gemm_f32", "class_out", "output_0"]
OUT_LANDMARK_ALIASES = ["landmark_out_Gemm_f32", "landmark_out", "output_2"]
RECOG_OUT_ALIASES    = ["embedding_Div_f32", "embedding_Div", "embedding",
                        "embedding_LpNormalization_f32", "output", "output_0"]

IMG_MEAN   = [0.0, 0.0, 0.0]
IMG_SCALE  = [0.0039215686, 0.0039215686, 0.0039215686]
RECOG_MEAN = [127.5, 127.5, 127.5]
RECOG_SCALE = [0.0078125, 0.0078125, 0.0078125]

# =====================================================================
# RUNTIME SETTINGS
# =====================================================================
CELEBA_W     = 178
CELEBA_H     = 218
CROP_W       = CELEBA_W
CROP_H       = CELEBA_H
EYE_V_OFFSET = 0.51
MODEL_W      = 224
MODEL_H      = 224
RECOG_W      = 112
RECOG_H      = 112

DETECT_CONF  = 0.40
DETECT_IOU   = 0.45
LM_THRESH    = 0.40
LM_ALPHA     = 0.35
RECOG_THRESH = CFG.get("match_threshold", 0.045)
REGISTER_FRAMES = 7

LM_NAMES  = ["LE", "RE", "N", "LM", "RM"]
LM_COLORS = [
    image.COLOR_RED, image.COLOR_BLUE, image.COLOR_GREEN,
    image.COLOR_YELLOW, image.COLOR_WHITE,
]


# =====================================================================
# SMALL UTILITIES (unchanged from v1)
# =====================================================================
def get_tensor_array(outputs, aliases):
    for name in aliases:
        try:
            value = outputs.get_tensor(name)
            if value is not None:
                return tensor.tensor_to_numpy_float32(value).flatten()
        except Exception:
            pass
    return None


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))


def l2_normalize(vec):
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-12:
        return vec
    return [v / norm for v in vec]


def cosine_distance(a, b):
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    return 1.0 - dot


def average_embeddings(embeddings):
    if not embeddings:
        return None
    dim = len(embeddings[0])
    avg = [0.0] * dim
    for emb in embeddings:
        for i in range(dim):
            avg[i] += emb[i]
    inv = 1.0 / len(embeddings)
    return l2_normalize([v * inv for v in avg])


# =====================================================================
# LOCAL FACE DB (legacy local registration — kept for dev/testing)
# =====================================================================
def load_face_db():
    if not os.path.exists(FACE_DB_PATH):
        return {}
    try:
        with open(FACE_DB_PATH, "r") as f:
            data = json.load(f)
        db = {}
        for name, emb in data.items():
            if isinstance(emb, list) and len(emb) > 0:
                db[name] = l2_normalize([float(v) for v in emb])
        print("Loaded local face DB: {} identities".format(len(db)))
        return db
    except Exception as e:
        print("Face DB load failed:", e)
        return {}


def save_face_db(db):
    try:
        with open(FACE_DB_PATH, "w") as f:
            json.dump(db, f)
        print("Saved face DB:", len(db))
    except Exception as e:
        print("Face DB save failed:", e)


def read_register_request():
    if not os.path.exists(REGISTER_REQUEST_PATH):
        return None
    try:
        with open(REGISTER_REQUEST_PATH, "r") as f:
            name = f.read().strip()
        os.remove(REGISTER_REQUEST_PATH)
        return name or "Person"
    except Exception as e:
        print("Register request read failed:", e)
        return None


def handle_clear_request(db):
    if not os.path.exists(CLEAR_DB_FLAG_PATH):
        return False
    try:
        os.remove(CLEAR_DB_FLAG_PATH)
    except Exception:
        pass
    db.clear()
    save_face_db(db)
    print("Cleared face DB")
    return True


# =====================================================================
# IMAGE HELPERS (unchanged from v1)
# =====================================================================
def make_crop_with_padding(frame, crop_x, crop_y, crop_w, crop_h):
    src_x1 = max(0, crop_x)
    src_y1 = max(0, crop_y)
    src_x2 = min(frame.width(), crop_x + crop_w)
    src_y2 = min(frame.height(), crop_y + crop_h)

    padded = image.Image(crop_w, crop_h, image.Format.FMT_RGB888)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return padded

    valid_crop = frame.crop(src_x1, src_y1, src_x2 - src_x1, src_y2 - src_y1)
    dst_x = max(0, -crop_x)
    dst_y = max(0, -crop_y)
    padded.draw_image(dst_x, dst_y, valid_crop)
    return padded


def resize_image(src, width, height):
    try:
        return src.resize(width, height)
    except Exception:
        dst = image.Image(width, height, image.Format.FMT_RGB888)
        dst.draw_image(0, 0, src)
        return dst


def make_v9_input(face_crop):
    canvas = image.Image(MODEL_W, MODEL_H, image.Format.FMT_RGB888)
    try:
        resized = resize_image(face_crop, MODEL_W, MODEL_H)
        canvas.draw_image(0, 0, resized)
    except Exception:
        canvas.draw_image(0, 0, face_crop)
    return canvas


def make_recognition_crop(frame, lm_abs):
    xs = [lm_abs[i * 2]     for i in range(5)]
    ys = [lm_abs[i * 2 + 1] for i in range(5)]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    center_x = int(sum(xs) / 5.0)
    center_y = int(sum(ys) / 5.0)

    face_w = max_x - min_x
    face_h = max_y - min_y
    side   = int(max(face_w * 2.4, face_h * 2.0, 64))
    crop_x = int(center_x - side / 2)
    crop_y = int(center_y - side * 0.45)

    crop = make_crop_with_padding(frame, crop_x, crop_y, side, side)
    return resize_image(crop, RECOG_W, RECOG_H)


def extract_embedding(aligned_face):
    outputs = RECOG_MODEL.forward_image(
        aligned_face, RECOG_MEAN, RECOG_SCALE,
        image.Fit.FIT_FILL, True, False
    )
    if not outputs:
        return None
    arr = get_tensor_array(outputs, RECOG_OUT_ALIASES)
    if arr is None:
        return None
    return l2_normalize([float(v) for v in arr])


def match_identity_local_db(embedding, db):
    """Legacy: match against local face_db.json (dev/testing mode)."""
    best_name = "Unknown"
    best_dist = 999.0
    for name, reg_emb in db.items():
        dist = cosine_distance(embedding, reg_emb)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    if best_dist >= RECOG_THRESH:
        best_name = "Unknown"
    return best_name, best_dist


# =====================================================================
# SYNC LOOP STATE
# =====================================================================
SYNC_FLAG_PATH = "/root/sync_now.flag"


def check_sync_flag() -> bool:
    """Returns True and removes flag if /root/sync_now.flag exists."""
    if os.path.exists(SYNC_FLAG_PATH):
        try:
            os.remove(SYNC_FLAG_PATH)
        except Exception:
            pass
        return True
    return False


def do_sync(active_flight_id: int) -> int:
    """Sync cache for active flight + any configured flights."""
    flight_ids = list(set(CFG.get("flight_ids", [1]) + [active_flight_id]))
    total = 0
    for fid in flight_ids:
        n = cache_mgr.sync(fid)
        if n >= 0:
            total += n
    return total


# =====================================================================
# MAIN
# =====================================================================
def main():
    cam_w = FACE_DET.input_width()
    cam_h = FACE_DET.input_height()
    cam  = camera.Camera(cam_w, cam_h, FACE_DET.input_format())
    disp = display.Display()

    print("Display: {}x{}".format(disp.width(), disp.height()))
    print("Camera:  {}x{}".format(cam_w, cam_h))
    print("Server:  {}".format(CFG["server_url"]))
    print("Pipeline: YOLO -> v9 landmarks -> ArcFace P3 -> cache match")
    print("Sync trigger: touch /root/sync_now.flag")
    print("Change flight: echo ID > /root/active_flight.txt")

    # Startup: initial cache sync
    active_flight = read_active_flight()
    draw_status_msg = "Syncing flight {}...".format(active_flight)
    print(draw_status_msg)
    synced = do_sync(active_flight)
    print("[main] Initial sync: {} embeddings".format(synced))

    face_db = load_face_db()  # Legacy local DB (dev testing)
    ema_lm = None
    pending_name = None
    pending_embeddings = []
    frame_idx = 0
    last_sync_time = time.time()
    sync_interval = CFG.get("sync_interval_sec", 300)
    last_result = None  # Cache last match result to persist overlay

    while not app.need_exit():
        img = cam.read()

        # --- Read active flight (can change at runtime) ---
        active_flight = read_active_flight()

        # --- Periodic auto-sync ---
        now = time.time()
        if (now - last_sync_time) >= sync_interval or check_sync_flag():
            print("[main] Auto-sync triggered")
            do_sync(active_flight)
            last_sync_time = now

        # --- Legacy local DB controls ---
        handle_clear_request(face_db)
        request_name = read_register_request()
        if request_name:
            pending_name = request_name
            pending_embeddings = []
            print("Registering '{}' with {} frames".format(pending_name, REGISTER_FRAMES))

        # --- Face detection ---
        objs = FACE_DET.detect(img, conf_th=DETECT_CONF, iou_th=DETECT_IOU)

        if len(objs) == 0:
            ema_lm = None
            last_result = None
            draw_status(img, "No face", image.COLOR_RED)
        else:
            for obj in objs:
                x, y, w, h = int(obj.x), int(obj.y), int(obj.w), int(obj.h)
                crop_x = int(x + w / 2 - CROP_W / 2)
                crop_y = int(y + h * 0.4 - CROP_H * EYE_V_OFFSET)
                face_crop = make_crop_with_padding(img, crop_x, crop_y, CROP_W, CROP_H)
                canvas = make_v9_input(face_crop)

                img.draw_rect(x, y, w, h, color=image.COLOR_GREEN, thickness=2)
                img.draw_rect(crop_x, crop_y, CROP_W, CROP_H,
                              color=image.Color(255, 200, 0), thickness=1)

                outputs = LM_MODEL.forward_image(
                    canvas, IMG_MEAN, IMG_SCALE,
                    image.Fit.FIT_FILL, True, False
                )
                if not outputs:
                    continue

                class_arr    = get_tensor_array(outputs, OUT_CLASS_ALIASES)
                landmark_arr = get_tensor_array(outputs, OUT_LANDMARK_ALIASES)
                if class_arr is None or landmark_arr is None:
                    img.draw_string(x, max(0, y - 15), "v9 miss", image.COLOR_RED)
                    continue

                score = sigmoid(class_arr[0])
                landmark_arr = [max(0.0, min(1.0, float(v))) for v in landmark_arr]

                if ema_lm is None:
                    ema_lm = landmark_arr[:]
                else:
                    ema_lm = [LM_ALPHA * landmark_arr[i] + (1 - LM_ALPHA) * ema_lm[i]
                              for i in range(10)]

                if score <= LM_THRESH:
                    img.draw_string(x, max(0, y - 15),
                                    "low:{:.2f}".format(score), image.COLOR_RED)
                    continue

                lm_abs = []
                for i in range(5):
                    lx = int(ema_lm[i * 2] * CROP_W) + crop_x
                    ly = int(ema_lm[i * 2 + 1] * CROP_H) + crop_y
                    lx = max(0, min(cam_w - 1, lx))
                    ly = max(0, min(cam_h - 1, ly))
                    lm_abs.append(lx)
                    lm_abs.append(ly)
                    img.draw_circle(lx, ly, 3, LM_COLORS[i], -1)

                recog_face = make_recognition_crop(img, lm_abs)
                embedding  = extract_embedding(recog_face)
                if embedding is None:
                    img.draw_string(x, max(0, y - 15), "P3 fail", image.COLOR_RED)
                    continue

                # --- Legacy local registration (dev mode) ---
                if pending_name:
                    pending_embeddings.append(embedding)
                    progress = len(pending_embeddings)
                    img.draw_string(10, 50,
                                    "Register {} {}/{}".format(
                                        pending_name, progress, REGISTER_FRAMES),
                                    image.COLOR_YELLOW)
                    if progress >= REGISTER_FRAMES:
                        face_db[pending_name] = average_embeddings(pending_embeddings)
                        save_face_db(face_db)
                        print("Registered:", pending_name)
                        pending_name = None
                        pending_embeddings = []

                # --- PRIMARY: Cache + Server match (GD4) ---
                cache_info = cache_mgr.get_cache_info(active_flight)
                if cache_info.get("cached"):
                    last_result = cache_mgr.match(embedding, active_flight)
                    draw_match_result(img, last_result, x, y, w, h)
                else:
                    # No cache yet — fall back to legacy local DB
                    identity, dist = match_identity_local_db(embedding, face_db)
                    if identity == "Unknown":
                        draw_no_match(img, x, y, w, h)
                    else:
                        img.draw_rect(x, y, w, h,
                                      color=image.COLOR_GREEN, thickness=3)
                        img.draw_string(x, max(0, y - 18),
                                        "{} (local) d={:.3f}".format(identity, dist),
                                        image.COLOR_GREEN)

                img.draw_string(x, y + h + 3,
                                "v9:{:.2f}".format(score), image.COLOR_YELLOW)
                break  # Process first face only

        # --- HUD ---
        cache_info = cache_mgr.get_cache_info(active_flight)
        draw_hud(img,
                 db_count=len(face_db),
                 cache_count=cache_info.get("count", 0),
                 flight_id=active_flight,
                 threshold=RECOG_THRESH)

        disp.show(img)

        frame_idx += 1
        if frame_idx % 30 == 0:
            gc.collect()


if __name__ == "__main__":
    main()
