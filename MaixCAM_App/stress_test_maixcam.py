#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIoT MaixCAM Stress Test - Do hieu nang trich xuat vector tren thiet bi bien
Test cac thanh phan: YOLO, V9, ArcFace P3, Pipeline E2E
"""
import io
import os
import platform
import sys
import time
import gc
import random
import math

# Conditional import - maix only exists on MaixCAM device
if platform.system() == "Linux":
    from maix import camera, display, image, nn, app, tensor
else:
    nn = None  # Placeholder for PC simulation


# =====================================================================
# CONFIG
# =====================================================================
MODEL_DIR = "/root/models"
SERVER_URL = "http://10.0.0.1:8000"

IS_MAIXCAM = platform.system() == "Linux" and os.path.exists("/root/models")

if IS_MAIXCAM:
    if os.path.exists(MODEL_DIR + "/yolo11s_face.mud"):
        FACE_DET = nn.YOLO11(model=MODEL_DIR + "/yolo11s_face.mud", dual_buff=False)
    else:
        FACE_DET = nn.YOLOv8(model=MODEL_DIR + "/yolov8n_face.mud", dual_buff=False)
    LM_MODEL    = nn.NN(MODEL_DIR + "/face_detect_v9.mud")
    RECOG_MODEL = nn.NN(MODEL_DIR + "/face_recognize_arcface_p3.mud")

OUT_CLASS_ALIASES     = ["class_out_Gemm_f32", "class_out", "output_0"]
OUT_LANDMARK_ALIASES  = ["landmark_out_Gemm_f32", "landmark_out", "output_2"]
RECOG_OUT_ALIASES     = ["embedding_Div_f32", "embedding_Div", "embedding",
                         "embedding_LpNormalization_f32", "output", "output_0"]

IMG_MEAN    = [0.0, 0.0, 0.0]
IMG_SCALE   = [0.0039215686, 0.0039215686, 0.0039215686]
RECOG_MEAN  = [127.5, 127.5, 127.5]
RECOG_SCALE = [0.0078125, 0.0078125, 0.0078125]

# Runtime settings
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

BAR_WIDTH = 40


# =====================================================================
# PROGRESS BAR
# =====================================================================
class ProgressBar:
    def __init__(self, total: int, prefix: str = ""):
        self.total = total
        self.prefix = prefix
        self.current = 0
        self.start_time = time.time()
        self._last_line_len = 0
    
    def update(self, current: int = None, suffix: str = ""):
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        percent = self.current / self.total if self.total > 0 else 0
        filled = int(BAR_WIDTH * percent)
        bar = "=" * filled + "-" * (BAR_WIDTH - filled)
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = "ETA: {}s".format(int(eta))
        else:
            eta_str = "ETA: --s"
        
        line = "\r{} [{}] {:5.1f}% {} {}".format(
            self.prefix, bar, percent*100, eta_str, suffix)
        spaces = " " * max(0, self._last_line_len - len(line))
        sys.stdout.write(line + spaces + "\r")
        sys.stdout.flush()
        self._last_line_len = len(line)
    
    def finish(self, message: str = ""):
        elapsed = time.time() - self.start_time
        bar = "=" * BAR_WIDTH
        line = "\r{} [{}] 100.0% {} ({:.1f}s)\n".format(
            self.prefix, bar, message, elapsed)
        spaces = " " * max(0, self._last_line_len - len(line))
        sys.stdout.write(line + spaces)
        sys.stdout.flush()


# =====================================================================
# UTILITIES
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
    
    src_x1 = max(0, crop_x)
    src_y1 = max(0, crop_y)
    src_x2 = min(frame.width(), crop_x + side)
    src_y2 = min(frame.height(), crop_y + side)
    
    padded = image.Image(side, side, image.Format.FMT_RGB888)
    if src_x2 > src_x1 and src_y2 > src_y1:
        valid_crop = frame.crop(src_x1, src_y1, src_x2 - src_x1, src_y2 - src_y1)
        dst_x = max(0, -crop_x)
        dst_y = max(0, -crop_y)
        padded.draw_image(dst_x, dst_y, valid_crop)
    
    return resize_image(padded, RECOG_W, RECOG_H)


# =====================================================================
# PIPELINE STAGES
# =====================================================================
def run_yolo_detection(img):
    """YOLO face detection"""
    t0 = time.time()
    objs = FACE_DET.detect(img, conf_th=DETECT_CONF, iou_th=DETECT_IOU)
    t1 = time.time()
    return objs, (t1 - t0) * 1000  # ms


def run_v9_landmarks(crop, canvas):
    """V9 landmark extraction"""
    t0 = time.time()
    outputs = LM_MODEL.forward_image(canvas, IMG_MEAN, IMG_SCALE,
                                     image.Fit.FIT_FILL, True, False)
    t1 = time.time()
    
    if not outputs:
        return None, None, (t1 - t0) * 1000
    
    class_arr    = get_tensor_array(outputs, OUT_CLASS_ALIASES)
    landmark_arr = get_tensor_array(outputs, OUT_LANDMARK_ALIASES)
    
    if class_arr is None or landmark_arr is None:
        return None, None, (t1 - t0) * 1000
    
    landmark_arr = [max(0.0, min(1.0, float(v))) for v in landmark_arr]
    return class_arr, landmark_arr, (t1 - t0) * 1000


def run_arcface_p3(aligned_face):
    """ArcFace P3 embedding extraction"""
    t0 = time.time()
    outputs = RECOG_MODEL.forward_image(
        aligned_face, RECOG_MEAN, RECOG_SCALE,
        image.Fit.FIT_FILL, True, False
    )
    t1 = time.time()
    
    if not outputs:
        return None, (t1 - t0) * 1000
    
    arr = get_tensor_array(outputs, RECOG_OUT_ALIASES)
    if arr is None:
        return None, (t1 - t0) * 1000
    
    return l2_normalize([float(v) for v in arr]), (t1 - t0) * 1000


# =====================================================================
# STRESS TEST
# =====================================================================
def run_stress_test(num_iterations: int, use_camera: bool = True):
    """Run stress test on MaixCAM"""
    
    print("=" * 60)
    print("  MAIXCAM STRESS TEST - Vector Extraction Performance")
    print("=" * 60)
    print()
    print("Configuration:")
    print("  - Iterations: {}".format(num_iterations))
    print("  - Mode: {}".format("Camera" if use_camera else "Synthetic"))
    print()
    
    # Init camera if using real input
    if use_camera:
        print("[INIT] Starting camera...")
        cam_w = FACE_DET.input_width()
        cam_h = FACE_DET.input_height()
        cam = camera.Camera(cam_w, cam_h, FACE_DET.input_format())
        print("[OK] Camera ready: {}x{}".format(cam_w, cam_h))
    
    # Timing storage
    yolo_times = []
    v9_times = []
    p3_times = []
    pipeline_times = []
    total_faces = 0
    
    progress = ProgressBar(num_iterations, "     ")
    
    for i in range(num_iterations):
        pipeline_start = time.time()
        
        if use_camera:
            # Read from camera
            img = cam.read()
        else:
            # Synthetic test (faster, for pure inference testing)
            img = None
        
        # Stage 1: YOLO Detection
        if use_camera and img:
            objs, yolo_ms = run_yolo_detection(img)
        else:
            objs, yolo_ms = [], 0.0
        
        yolo_times.append(yolo_ms)
        
        if use_camera and img and len(objs) > 0:
            obj = objs[0]
            x, y, w, h = int(obj.x), int(obj.y), int(obj.w), int(obj.h)
            
            crop_x = int(x + w / 2 - CROP_W / 2)
            crop_y = int(y + h * 0.4 - CROP_H * EYE_V_OFFSET)
            
            src_x1 = max(0, crop_x)
            src_y1 = max(0, crop_y)
            src_x2 = min(img.width(), crop_x + CROP_W)
            src_y2 = min(img.height(), crop_y + CROP_H)
            
            face_crop = image.Image(CROP_W, CROP_H, image.Format.FMT_RGB888)
            if src_x2 > src_x1 and src_y2 > src_y1:
                valid = img.crop(src_x1, src_y1, src_x2 - src_x1, src_y2 - src_y1)
                face_crop.draw_image(0, 0, valid)
            
            canvas = make_v9_input(face_crop)
            
            # Stage 2: V9 Landmarks
            class_arr, landmark_arr, v9_ms = run_v9_landmarks(face_crop, canvas)
            v9_times.append(v9_ms)
            
            if class_arr is not None and landmark_arr is not None:
                score = sigmoid(class_arr[0])
                
                if score > LM_THRESH:
                    lm_abs = []
                    for j in range(5):
                        lx = int(landmark_arr[j * 2] * CROP_W) + crop_x
                        ly = int(landmark_arr[j * 2 + 1] * CROP_H) + crop_y
                        lx = max(0, min(cam_w - 1, lx))
                        ly = max(0, min(cam_h - 1, ly))
                        lm_abs.append(lx)
                        lm_abs.append(ly)
                    
                    recog_face = make_recognition_crop(img, lm_abs)
                    
                    # Stage 3: ArcFace P3
                    embedding, p3_ms = run_arcface_p3(recog_face)
                    p3_times.append(p3_ms)
                    
                    if embedding is not None:
                        total_faces += 1
                else:
                    p3_times.append(0)
            else:
                v9_times[-1] = 0
                p3_times.append(0)
        else:
            v9_times.append(0)
            p3_times.append(0)
        
        pipeline_end = time.time()
        pipeline_times.append((pipeline_end - pipeline_start) * 1000)
        
        progress.update(suffix="Iter {}/{}".format(i + 1, num_iterations))
        
        # Periodic GC
        if (i + 1) % 30 == 0:
            gc.collect()
    
    progress.finish("Done")
    
    # Calculate statistics
    def calc_stats(times):
        if not times:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}
        valid = [t for t in times if t > 0]
        if not valid:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}
        valid_sorted = sorted(valid)
        return {
            "count": len(valid),
            "avg": sum(valid) / len(valid),
            "min": min(valid),
            "max": max(valid),
            "p50": valid_sorted[len(valid_sorted) // 2],
            "p95": valid_sorted[int(len(valid_sorted) * 0.95)] if len(valid_sorted) > 1 else valid_sorted[0],
            "p99": valid_sorted[int(len(valid_sorted) * 0.99)] if len(valid_sorted) > 2 else valid_sorted[-1],
        }
    
    yolo_stats = calc_stats(yolo_times)
    v9_stats = calc_stats(v9_times)
    p3_stats = calc_stats(p3_times)
    pipe_stats = calc_stats(pipeline_times)
    
    return {
        "total_iterations": num_iterations,
        "total_faces": total_faces,
        "yolo": yolo_stats,
        "v9": v9_stats,
        "p3": p3_stats,
        "pipeline": pipe_stats,
    }


def print_results(results):
    """Print stress test results"""
    print()
    print("=" * 60)
    print("  STRESS TEST RESULTS")
    print("=" * 60)
    print()
    print("Summary:")
    print("  Total iterations: {}".format(results["total_iterations"]))
    print("  Faces detected:  {} ({:.1f}%)".format(
        results["total_faces"],
        results["total_faces"] / results["total_iterations"] * 100))
    print()
    
    print("-" * 60)
    print("  STAGE           |    AVG    |    P50    |    P95    |    P99")
    print("-" * 60)
    
    stages = [
        ("YOLO Detection", results["yolo"]),
        ("V9 Landmarks", results["v9"]),
        ("ArcFace P3", results["p3"]),
        ("Pipeline E2E", results["pipeline"]),
    ]
    
    for name, stats in stages:
        if stats["count"] > 0:
            print("  {:16} | {:8.2f}ms | {:8.2f}ms | {:8.2f}ms | {:8.2f}ms".format(
                name, stats["avg"], stats["p50"], stats["p95"], stats["p99"]))
        else:
            print("  {:16} |    N/A    |    N/A    |    N/A    |    N/A".format(name))
    
    print("-" * 60)
    print()
    
    # Throughput calculation
    p3_stats = results["p3"]
    if p3_stats["count"] > 0:
        fps = 1000.0 / p3_stats["avg"]
        print("Throughput (ArcFace P3): {:.1f} embeddings/second".format(fps))
        print()
    
    # Pass/Fail criteria
    print("Evaluation:")
    p95_pipeline = results["pipeline"]["p95"]
    if p95_pipeline < 500:
        print("  [PASS] E2E P95 < 500ms")
    else:
        print("  [FAIL] E2E P95 >= 500ms")
    
    if results["total_faces"] / results["total_iterations"] > 0.5:
        print("  [PASS] Face detection rate > 50%")
    else:
        print("  [WARN] Face detection rate < 50%")


# =====================================================================
# MAIN
# =====================================================================
def main():
    import sys
    
    # Parse arguments
    num_iterations = 100
    use_camera = True
    
    if len(sys.argv) > 1:
        try:
            num_iterations = int(sys.argv[1])
        except ValueError:
            print("Usage: python stress_test_maixcam.py [iterations]")
            return
    
    print()
    print("[INFO] Starting MaixCAM Stress Test")
    print("[INFO] Iterations: {}".format(num_iterations))
    print("[INFO] Press Ctrl+C to stop early")
    print()
    
    try:
        results = run_stress_test(num_iterations, use_camera)
        print_results(results)
        
        print()
        print("[SUCCESS] Stress test completed!")
        
    except KeyboardInterrupt:
        print()
        print("[INFO] Interrupted by user")
    except Exception as e:
        print()
        print("[ERROR] {}".format(e))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
