from maix import camera, display, image, nn, app, sys, tensor
import gc
import json
import math
import os
import time

from config import load_config
from sync_cache import CacheManager
from display import (
    draw_match_result, draw_status, draw_hud, draw_no_match,
    draw_target_brackets, draw_string_ascii, COLOR_PRIMARY, COLOR_ACCENT,
    COLOR_OK, COLOR_DANGER, COLOR_WARN, COLOR_TEXT, COLOR_MUTED, COLOR_BLACK
)
import mjpeg_server

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

import platform
IS_MAIXCAM2 = platform.machine() == "rv1126"
if IS_MAIXCAM2:
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
CROP_W       = CELEBA_W * 2
CROP_H       = CELEBA_H * 2
EYE_V_OFFSET = 0.51
MODEL_W      = 224
MODEL_H      = 224
RECOG_W      = 112
RECOG_H      = 112

DETECT_CONF  = 0.40
DETECT_IOU   = 0.45
LM_THRESH    = 0.40
LM_ALPHA     = 0.35
RECOG_THRESH = CFG.get("match_threshold", 0.018)
REGISTER_FRAMES = 7
AI_FRAME_INTERVAL = max(1, int(CFG.get("ai_frame_interval", 3)))
RECOG_FRAME_INTERVAL = max(1, int(CFG.get("recognition_frame_interval", 9)))
OVERLAY_CACHE_TTL_SEC = float(CFG.get("overlay_cache_ttl_sec", 1.0))
RECOG_CACHE_TTL_SEC = float(CFG.get("recognition_cache_ttl_sec", 1.5))

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


def draw_cached_overlay(img, overlay, now):
    if overlay is None:
        return False
    if now - overlay.get("time", 0.0) > OVERLAY_CACHE_TTL_SEC:
        return False

    x = overlay["x"]
    y = overlay["y"]
    w = overlay["w"]
    h = overlay["h"]
    crop_x = overlay["crop_x"]
    crop_y = overlay["crop_y"]
    lm_abs = overlay.get("lm_abs", [])

    mode = overlay.get("mode")
    
    # Draw target brackets based on status
    if mode == "match":
        draw_target_brackets(img, x, y, w, h, COLOR_OK, thickness=3)
    elif mode == "no_match":
        draw_target_brackets(img, x, y, w, h, COLOR_DANGER, thickness=3)
    else:
        draw_target_brackets(img, x, y, w, h, COLOR_PRIMARY, thickness=2)

    # Faint outer crop area guide box
    img.draw_rect(crop_x, crop_y, CROP_W, CROP_H,
                  color=COLOR_MUTED, thickness=1)

    # Draw Face Mesh landmarks & wireframe
    num_pts = min(5, int(len(lm_abs) / 2))
    for i in range(num_pts):
        lx = lm_abs[i * 2]
        ly = lm_abs[i * 2 + 1]
        img.draw_circle(lx, ly, 3, COLOR_ACCENT, -1)
    
    if num_pts == 5:
        connections = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]
        for start_idx, end_idx in connections:
            x1 = lm_abs[start_idx * 2]
            y1 = lm_abs[start_idx * 2 + 1]
            x2 = lm_abs[end_idx * 2]
            y2 = lm_abs[end_idx * 2 + 1]
            try:
                img.draw_line(x1, y1, x2, y2, color=COLOR_MUTED, thickness=1)
            except Exception:
                pass

    if mode == "match":
        draw_match_result(img, overlay.get("result"), x, y, w, h)
    elif mode == "local":
        draw_target_brackets(img, x, y, w, h, COLOR_OK, thickness=3)
        draw_string_ascii(img, x, max(0, y - 18), overlay.get("label", ""),
                        COLOR_OK)
    elif mode == "no_match":
        draw_no_match(img, x, y, w, h)

    score = overlay.get("score")
    if score is not None:
        draw_string_ascii(img, x, y + h + 3, "v9:{:.2f}".format(score),
                        COLOR_WARN)
    return True


def apply_cached_recognition(overlay, recognition, now):
    if recognition is None:
        return False
    if now - recognition.get("time", 0.0) > RECOG_CACHE_TTL_SEC:
        return False

    overlay["mode"] = recognition.get("mode")
    if "result" in recognition:
        overlay["result"] = recognition.get("result")
    if "label" in recognition:
        overlay["label"] = recognition.get("label")
    return True


def draw_recognition_overlay(img, overlay):
    mode = overlay.get("mode")
    if not mode:
        return

    x = overlay["x"]
    y = overlay["y"]
    w = overlay["w"]
    h = overlay["h"]
    if mode == "match":
        draw_match_result(img, overlay.get("result"), x, y, w, h)
    elif mode == "local":
        draw_target_brackets(img, x, y, w, h, COLOR_OK, thickness=3)
        draw_string_ascii(img, x, max(0, y - 18), overlay.get("label", ""),
                        COLOR_OK)
    elif mode == "no_match":
        draw_no_match(img, x, y, w, h)


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
    cam_w = 640
    cam_h = 480
    cam  = camera.Camera(cam_w, cam_h, FACE_DET.input_format())

    enable_lcd = CFG.get("enable_lcd", True)
    disp = None
    if enable_lcd:
        try:
            disp = display.Display()
            print("Display: {}x{}".format(disp.width(), disp.height()))
        except Exception as e:
            print("Failed to initialize LCD display:", e)
            disp = None
    else:
        print("LCD Display is disabled in config")

    mjpeg_server.configure(
        CFG.get("stream_fps_limit", 15),
        CFG.get("stream_jpeg_quality", 70)
    )
    mjpeg_server.start_server(8080)
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
    overlay_cache = None
    recognition_cache = None
    no_face_frames = 0  # Hysteresis frame counter to prevent HUD flickering
    # Throttle file I/O: only check disk-based flags/files at most once per second
    IO_CHECK_INTERVAL = 1.0
    last_io_check = 0.0
    active_flight = read_active_flight()  # Read once at startup

    # Freeze state variables for successful match timeout
    freeze_active = False
    freeze_start_time = 0.0
    frozen_face_img = None
    frozen_result = None
    frozen_x, frozen_y, frozen_w, frozen_h = 0, 0, 0, 0

    while not app.need_exit():
        img = cam.read()
        now = time.time()

        # Handle successful match freeze state
        if freeze_active:
            if now - freeze_start_time >= 5.0:
                # Timeout elapsed: reset and return to normal scanning
                freeze_active = False
                frozen_face_img = None
                frozen_result = None
                ema_lm = None
                last_result = None
                recognition_cache = None
                overlay_cache = None
            else:
                # Draw green brackets around the face's frozen location
                draw_target_brackets(img, frozen_x, frozen_y, frozen_w, frozen_h, COLOR_OK, thickness=3)

                # Draw boarding pass details
                draw_match_result(img, frozen_result, frozen_x, frozen_y, frozen_w, frozen_h)

                # Show remaining seconds countdown
                remaining = int(5.0 - (now - freeze_start_time) + 0.9)
                draw_status(img, "MATCHED - RESETS IN {}S".format(remaining), COLOR_OK)

                # Draw bottom HUD bar
                cache_info = cache_mgr.get_cache_info(active_flight)
                draw_hud(img,
                         db_count=len(face_db),
                         cache_count=cache_info.get("count", 0),
                         flight_id=active_flight,
                         threshold=RECOG_THRESH,
                         flight_status=cache_info.get("status"))

                if disp:
                    disp.show(img)
                mjpeg_server.update_frame(img)
                time.sleep(0.005)
                frame_idx += 1
                continue

        # --- Throttled file I/O: run at most once per IO_CHECK_INTERVAL seconds ---
        if now - last_io_check >= IO_CHECK_INTERVAL:
            last_io_check = now

            # Read active flight (can change at runtime via file)
            active_flight = read_active_flight()

            # Periodic auto-sync
            if (now - last_sync_time) >= sync_interval or check_sync_flag():
                print("[main] Auto-sync triggered")
                do_sync(active_flight)
                last_sync_time = now

            # Legacy local DB controls
            handle_clear_request(face_db)
            request_name = read_register_request()
            if request_name:
                pending_name = request_name
                pending_embeddings = []
                print("Registering '{}' with {} frames".format(pending_name, REGISTER_FRAMES))

        should_run_ai = (frame_idx % AI_FRAME_INTERVAL == 0)
        if not should_run_ai:
            if not draw_cached_overlay(img, overlay_cache, now):
                overlay_cache = None
        else:
            # --- Face detection ---
            objs = FACE_DET.detect(img, conf_th=DETECT_CONF, iou_th=DETECT_IOU)

            if len(objs) == 0:
                no_face_frames += 1
                if no_face_frames >= 12:  # Grace period of 12 loops (~800ms) before clearing
                    ema_lm = None
                    last_result = None
                    recognition_cache = None
                    overlay_cache = None
                    draw_status(img, "No face", COLOR_DANGER)
                else:
                    # Keep showing cached overlay during grace period
                    if not draw_cached_overlay(img, overlay_cache, now):
                        overlay_cache = None
                        draw_status(img, "No face", COLOR_DANGER)
            else:
                no_face_frames = 0
                for obj in objs:
                    x, y, w, h = int(obj.x), int(obj.y), int(obj.w), int(obj.h)
                    crop_x = int(x + w / 2 - CROP_W / 2)
                    crop_y = int(y + h * 0.4 - CROP_H * EYE_V_OFFSET)
                    face_crop = make_crop_with_padding(img, crop_x, crop_y, CROP_W, CROP_H)
                    canvas = make_v9_input(face_crop)

                    # Dynamic brackets while identifying (laser scanline removed)
                    draw_target_brackets(img, x, y, w, h, COLOR_PRIMARY, thickness=3)
                    img.draw_rect(crop_x, crop_y, CROP_W, CROP_H,
                                  color=COLOR_MUTED, thickness=1)

                    outputs = LM_MODEL.forward_image(
                        canvas, IMG_MEAN, IMG_SCALE,
                        image.Fit.FIT_FILL, True, False
                    )
                    if not outputs:
                        continue

                    class_arr    = get_tensor_array(outputs, OUT_CLASS_ALIASES)
                    landmark_arr = get_tensor_array(outputs, OUT_LANDMARK_ALIASES)
                    if class_arr is None or landmark_arr is None:
                        draw_string_ascii(img, x, max(0, y - 15), "v9 miss", COLOR_DANGER)
                        continue

                    score = sigmoid(class_arr[0])
                    landmark_arr = [max(0.0, min(1.0, float(v))) for v in landmark_arr]

                    if ema_lm is None:
                        ema_lm = landmark_arr[:]
                    else:
                        ema_lm = [LM_ALPHA * landmark_arr[i] + (1 - LM_ALPHA) * ema_lm[i]
                                  for i in range(10)]

                    if score <= LM_THRESH:
                        draw_string_ascii(img, x, max(0, y - 15),
                                        "low:{:.2f}".format(score), COLOR_DANGER)
                        continue

                    lm_abs = []
                    for i in range(5):
                        lx = int(ema_lm[i * 2] * CROP_W) + crop_x
                        ly = int(ema_lm[i * 2 + 1] * CROP_H) + crop_y
                        lx = max(0, min(cam_w - 1, lx))
                        ly = max(0, min(cam_h - 1, ly))
                        lm_abs.append(lx)
                        lm_abs.append(ly)
                        img.draw_circle(lx, ly, 3, COLOR_ACCENT, -1)

                    # Draw Face Mesh connecting wireframe lines
                    connections = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]
                    for start_idx, end_idx in connections:
                        x1 = lm_abs[start_idx * 2]
                        y1 = lm_abs[start_idx * 2 + 1]
                        x2 = lm_abs[end_idx * 2]
                        y2 = lm_abs[end_idx * 2 + 1]
                        try:
                            img.draw_line(x1, y1, x2, y2, color=COLOR_MUTED, thickness=1)
                        except Exception:
                            pass

                    overlay_cache = {
                        "time": now,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "crop_x": crop_x,
                        "crop_y": crop_y,
                        "lm_abs": lm_abs,
                        "score": score,
                    }

                    should_run_recognition = (
                        pending_name is not None or
                        recognition_cache is None or
                        (now - recognition_cache.get("time", 0.0)) >= RECOG_CACHE_TTL_SEC or
                        (frame_idx % RECOG_FRAME_INTERVAL == 0)
                    )

                    if not should_run_recognition and apply_cached_recognition(
                            overlay_cache, recognition_cache, now):
                        draw_recognition_overlay(img, overlay_cache)
                    else:
                        recog_face = make_recognition_crop(img, lm_abs)
                        embedding  = extract_embedding(recog_face)
                        if embedding is None:
                            draw_string_ascii(img, x, max(0, y - 15), "P3 fail", COLOR_DANGER)
                        else:
                            # --- Legacy local registration (dev mode) ---
                            if pending_name:
                                pending_embeddings.append(embedding)
                                progress = len(pending_embeddings)
                                draw_string_ascii(img, 10, 50,
                                                "Register {} {}/{}".format(
                                                    pending_name, progress, REGISTER_FRAMES),
                                                COLOR_WARN)
                                if progress >= REGISTER_FRAMES:
                                    face_db[pending_name] = average_embeddings(pending_embeddings)
                                    save_face_db(face_db)
                                    print("Registered:", pending_name)
                                    pending_name = None
                                    pending_embeddings = []

                            # --- PRIMARY: Cache + Server match (GD4) ---
                            cache_info = cache_mgr.get_cache_info(active_flight)
                            recognition_cache = {"time": now}

                            # Check if recognition is allowed (flight not departed/arrived)
                            if not cache_info.get("recognition_allowed", True):
                                flight_status = cache_info.get("status", "unknown")
                                draw_status(img, "Flight {} - Recognition disabled".format(flight_status), COLOR_DANGER)
                            elif cache_info.get("cached"):
                                last_result = cache_mgr.match_local(embedding, active_flight)
                                overlay_cache["mode"] = "match"
                                overlay_cache["result"] = last_result
                                recognition_cache["mode"] = "match"
                                recognition_cache["result"] = last_result
                                draw_match_result(img, last_result, x, y, w, h)
                                
                                if last_result is not None:
                                    # Successful match: trigger 5-second freeze
                                    freeze_active = True
                                    freeze_start_time = now
                                    frozen_result = last_result
                                    frozen_face_img = recog_face
                                    frozen_x, frozen_y, frozen_w, frozen_h = x, y, w, h
                            else:
                                # No cache yet: fall back to legacy local DB
                                identity, dist = match_identity_local_db(embedding, face_db)
                                if identity == "Unknown":
                                    overlay_cache["mode"] = "no_match"
                                    recognition_cache["mode"] = "no_match"
                                    draw_no_match(img, x, y, w, h)
                                else:
                                    label = "{} (local) d={:.3f}".format(identity, dist)
                                    overlay_cache["mode"] = "local"
                                    overlay_cache["label"] = label
                                    recognition_cache["mode"] = "local"
                                    recognition_cache["label"] = label
                                    draw_target_brackets(img, x, y, w, h, COLOR_OK, thickness=3)
                                    draw_string_ascii(img, x, max(0, y - 18), label,
                                                    COLOR_OK)
                                    
                                    # Trigger 5-second freeze for local match too
                                    freeze_active = True
                                    freeze_start_time = now
                                    frozen_result = {
                                        "payload": {
                                            "passenger_name": identity,
                                            "gate": "LOCAL",
                                            "seat_number": "DB",
                                            "flight_number": "DEV_TEST",
                                            "dest_city": "LOCALHOST",
                                            "boarding_time": "NOW",
                                            "departure_time": "NOW"
                                        },
                                        "source": "local",
                                        "distance": dist
                                    }
                                    frozen_face_img = recog_face
                                    frozen_x, frozen_y, frozen_w, frozen_h = x, y, w, h

                    draw_string_ascii(img, x, y + h + 3,
                                    "v9:{:.2f}".format(score), COLOR_WARN)
                    break  # Process first face only

        # --- HUD ---
        cache_info = cache_mgr.get_cache_info(active_flight)
        draw_hud(img,
                 db_count=len(face_db),
                 cache_count=cache_info.get("count", 0),
                 flight_id=active_flight,
                 threshold=RECOG_THRESH,
                 flight_status=cache_info.get("status"))

        if disp:
            disp.show(img)
        mjpeg_server.update_frame(img)
        time.sleep(0.005)

        frame_idx += 1
        if frame_idx % 30 == 0:
            gc.collect()


if __name__ == "__main__":
    main()
