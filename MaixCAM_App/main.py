from maix import camera, display, image, nn, app, sys, tensor
import math

# =====================================================================
# CAU HINH MODEL
# =====================================================================
if sys.device_name().lower() == "maixcam2":
    FACE_DET = nn.YOLO11(model="/root/models/yolo11s_face.mud", dual_buff=False)
else:
    FACE_DET = nn.YOLOv8(model="/root/models/yolov8n_face.mud", dual_buff=False)

LM_MODEL = nn.NN("/root/models/face_detect_v3.mud")
OUT_CLASS    = "class_out_Gemm_f32"
OUT_BBOX     = "bbox_out_Gemm_f32"
OUT_LANDMARK = "landmark_out_Gemm_f32"

# forward_image: out = (pixel - mean) * scale.
IMG_MEAN  = [0.0, 0.0, 0.0]
IMG_SCALE = [0.0039215686, 0.0039215686, 0.0039215686]

# =====================================================================
# CROP SETTINGS
# =====================================================================
CELEBA_W = 178
CELEBA_H = 218
CROP_W = CELEBA_W
CROP_H = CELEBA_H

EYE_V_OFFSET = 0.51

# =====================================================================
# MODEL INPUT SIZE
# =====================================================================
MODEL_W = 224
MODEL_H = 224

LM_NAMES = ["LE", "RE", "N", "LM", "RM"]
LM_COLORS = [
    image.COLOR_RED, image.COLOR_BLUE, image.COLOR_GREEN,
    image.COLOR_YELLOW, image.COLOR_WHITE,
]

THRESH = 0.4
LM_ALPHA = 0.35

# =====================================================================
# DEBUG MODE: thay doi gia tri nay de thu nghiem
#   0 = khong flip
#   1 = flip trai-phai (hflip)  <-- thu nhieu nhat
#   2 = flip tren-duoi (vflip)
#   3 = ca hai flip
# =====================================================================
FLIP_MODE = 1


def make_crop_with_padding(frame: image.Image, crop_x: int, crop_y: int,
                           crop_w: int, crop_h: int) -> image.Image:
    """Crop face region with black padding."""
    src_x1 = max(0, crop_x)
    src_y1 = max(0, crop_y)
    src_x2 = min(frame.width(),  crop_x + crop_w)
    src_y2 = min(frame.height(), crop_y + crop_h)

    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return image.Image(crop_w, crop_h, image.Format.FMT_RGB888)

    valid_crop = frame.crop(src_x1, src_y1, src_x2 - src_x1, src_y2 - src_y1)
    padded = image.Image(crop_w, crop_h, image.Format.FMT_RGB888)
    dst_x = max(0, -crop_x)
    dst_y = max(0, -crop_y)
    padded.draw_image(dst_x, dst_y, valid_crop)
    return padded


def apply_flip(crop: image.Image, flip_mode: int) -> image.Image:
    """Apply flip to crop based on mode."""
    if flip_mode == 0:
        return crop
    elif flip_mode == 1:
        return crop.flip(1)   # horizontal
    elif flip_mode == 2:
        return crop.flip(0)   # vertical
    elif flip_mode == 3:
        return crop.flip(1).flip(0)
    return crop


def main():
    cam_w = FACE_DET.input_width()
    cam_h = FACE_DET.input_height()
    cam   = camera.Camera(cam_w, cam_h, FACE_DET.input_format())
    disp  = display.Display()

    print("Display: {}x{}".format(disp.width(), disp.height()))
    print("Camera:  {}x{}".format(cam_w, cam_h))
    print("Crop: {}x{}, Model: {}x{}".format(CROP_W, CROP_H, MODEL_W, MODEL_H))
    print("FLIP_MODE = {} (0=none, 1=hflip, 2=vflip, 3=both)".format(FLIP_MODE))
    print("Models loaded!")

    ema_lm = None
    frame  = 0

    while not app.need_exit():
        img = cam.read()

        objs = FACE_DET.detect(img, conf_th=0.4, iou_th=0.45)

        for obj in objs:
            x, y, w, h = int(obj.x), int(obj.y), int(obj.w), int(obj.h)

            crop_x = int(x + w / 2 - CROP_W / 2)
            crop_y = int(y + h * 0.4 - CROP_H * EYE_V_OFFSET)

            face_crop = make_crop_with_padding(img, crop_x, crop_y, CROP_W, CROP_H)

            # Apply flip experiment
            face_flip = apply_flip(face_crop, FLIP_MODE)

            # Stretch to model input
            canvas = image.Image(MODEL_W, MODEL_H, image.Format.FMT_RGB888)
            canvas.draw_image(0, 0, face_flip)

            # Draw overlays
            img.draw_rect(x, y, w, h, color=image.COLOR_GREEN, thickness=2)
            img.draw_rect(crop_x, crop_y, CROP_W, CROP_H,
                          color=image.Color(255, 200, 0), thickness=1)

            outputs = LM_MODEL.forward_image(
                canvas, IMG_MEAN, IMG_SCALE,
                image.Fit.FIT_FILL, True, False
            )

            if outputs:
                class_arr    = tensor.tensor_to_numpy_float32(outputs.get_tensor(OUT_CLASS)).flatten()
                landmark_arr = tensor.tensor_to_numpy_float32(outputs.get_tensor(OUT_LANDMARK)).flatten()

                score = 1.0 / (1.0 + math.exp(-float(class_arr[0])))
                landmark_arr = [max(0.0, min(1.0, v)) for v in landmark_arr]

                # EMA smoothing
                if ema_lm is None:
                    ema_lm = landmark_arr[:]
                else:
                    ema_lm = [LM_ALPHA * landmark_arr[i] + (1 - LM_ALPHA) * ema_lm[i]
                              for i in range(10)]

                if score > THRESH:
                    for i in range(5):
                        # Denormalize using CROP_W/H
                        lx = int(ema_lm[i * 2]     * CROP_W) + crop_x
                        ly = int(ema_lm[i * 2 + 1] * CROP_H) + crop_y
                        lx = max(0, min(cam_w - 1, lx))
                        ly = max(0, min(cam_h - 1, ly))
                        img.draw_circle(lx, ly, 3, LM_COLORS[i], -1)
                        img.draw_string(lx + 4, ly - 4, LM_NAMES[i], image.COLOR_WHITE)

                    label = "F:{:.2f} flip={}".format(score, FLIP_MODE)
                    img.draw_string(x, y - 15, label, image.COLOR_YELLOW)
                else:
                    img.draw_string(x, y - 15, "low:{:.2f}".format(score), image.COLOR_RED)

        if len(objs) == 0:
            ema_lm = None
            img.draw_string(10, 30, "No face", image.COLOR_RED)

        img.draw_string(10, 10, "AIoT FLD flip={}".format(FLIP_MODE), image.COLOR_GREEN)
        disp.show(img)
        frame += 1


if __name__ == "__main__":
    main()
