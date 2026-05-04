from maix import camera, display, image, nn, app, tensor
import numpy as np
import math

THRESH = 0.4

OUT_CLASS    = "class_out_Gemm_f32"
OUT_LANDMARK = "landmark_out_Gemm_f32"

IMG_MEAN  = [0.001, 0.001, 0.001]
IMG_SCALE = [0.003921568, 0.003921568, 0.003921568]

LM_NAMES = ["LE", "RE", "N", "LM", "RM"]
LM_COLORS = [
    image.COLOR_RED, image.COLOR_BLUE, image.COLOR_GREEN,
    image.COLOR_YELLOW, image.COLOR_WHITE,
]


def main():
    cam_w, cam_h = 320, 240
    cam  = camera.Camera(cam_w, cam_h)
    disp = display.Display()

    print(f"Display: {disp.width()}x{disp.height()}")
    print(f"Camera:  {cam_w}x{cam_h}")

    model = nn.NN("/root/models/face_detect_v3.mud")
    print("Model loaded!")

    frame = 0
    while not app.need_exit():
        img = cam.read()

        # Debug: danh dau goc tren-trai bang hinh vuong do
        img.draw_rect(0, 0, 30, 30, image.COLOR_RED, -1)
        img.draw_string(2, 2, "TL", image.COLOR_WHITE)
        # Danh dau tam anh
        img.draw_circle(cam_w // 2, cam_h // 2, 5, image.COLOR_WHITE, -1)

        try:
            outputs = model.forward_image(
                img, IMG_MEAN, IMG_SCALE,
                image.Fit.FIT_FILL, True, False
            )

            if outputs is None:
                disp.show(img)
                frame += 1
                continue

            class_arr    = tensor.tensor_to_numpy_float32(outputs.get_tensor(OUT_CLASS)).flatten()
            landmark_arr = tensor.tensor_to_numpy_float32(outputs.get_tensor(OUT_LANDMARK)).flatten()

            score = 1.0 / (1.0 + math.exp(-float(class_arr[0])))

            if score > THRESH:
                for i in range(5):
                    lx = int(landmark_arr[i * 2]     * cam_w)
                    ly = int(landmark_arr[i * 2 + 1] * cam_h)
                    lx = max(0, min(cam_w - 1, lx))
                    ly = max(0, min(cam_h - 1, ly))
                    img.draw_circle(lx, ly, 8, LM_COLORS[i], -1)
                    img.draw_string(lx + 10, ly - 5, LM_NAMES[i], image.COLOR_WHITE)

                    if frame < 3:
                        print(f"  {LM_NAMES[i]}: raw=({landmark_arr[i*2]:.3f},{landmark_arr[i*2+1]:.3f}) px=({lx},{ly})")

                img.draw_string(10, 35, f"Face:{score:.2f}", image.COLOR_YELLOW)

        except Exception as e:
            img.draw_string(2, 35, str(e)[:45], image.COLOR_RED)

        img.draw_string(10, 10, "AIoT FLD", image.COLOR_GREEN)
        disp.show(img)
        frame += 1


if __name__ == "__main__":
    main()
