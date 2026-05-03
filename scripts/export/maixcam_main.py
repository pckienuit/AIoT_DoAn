from maix import camera, display, image, nn, app
import time

def main():
    # 1. Khởi tạo Camera và Màn hình LCD
    # MaixCAM mặc định dùng độ phân giải này cho mượt
    cam = camera.Camera(320, 240)
    disp = display.Display()

    print("Đang nạp model NPU...")
    # 2. Nạp model .mud (copy file từ MaixVision vào thẻ nhớ MaixCAM)
    model = nn.Model("face_detect_model_v3.mud")
    print("Nạp model thành công!")

    # Lấy thông tin kích thước đầu vào của model (chắc chắn là 224x224)
    input_shape = model.inputs[0].shape
    MODEL_W = input_shape[2]  # 224
    MODEL_H = input_shape[3]  # 224

    while not app.need_exit():
        # Đọc khung hình từ camera
        img = cam.read()

        # 3. Tiền xử lý (Pre-processing)
        # NPU SG2002 yêu cầu ảnh đúng 224x224. Ta thu nhỏ ảnh camera lại.
        # Lưu ý: Việc chia 255.0 (scale) đã được NPU làm tự động nhờ cấu hình lúc convert trên MaixVision!
        img_resized = img.resize(MODEL_W, MODEL_H)

        # 4. Đưa vào NPU dự đoán (Inference)
        # Hàm forward trả về một list các tensor output (tuơng ứng class_out, bbox_out, landmark_out)
        t_start = time.time()
        outputs = model.forward(img_resized)
        t_infer = (time.time() - t_start) * 1000  # Tính ms

        # Thứ tự output phụ thuộc vào lúc export ONNX.
        # Ở đây ta giả sử: [0] = class, [1] = bbox, [2] = landmarks
        class_out = outputs[0].to_numpy()
        bbox_out = outputs[1].to_numpy()
        landmark_out = outputs[2].to_numpy()

        # Tính class score (Sigmoid)
        import math
        score = 1.0 / (1.0 + math.exp(-class_out[0][0]))

        # Nếu độ tự tin > 50% (có khuôn mặt)
        if score > 0.5:
            lm = landmark_out[0]
            
            # 5. Vẽ 5 điểm Landmarks (10 giá trị x, y)
            for i in range(0, 10, 2):
                # Tọa độ xuất ra từ model nằm trong khoảng [0, 1]
                # Ta cần map nó về kích thước của ảnh hiển thị (320x240)
                # Lưu ý: Vì ta nạp ảnh resize 224x224 trực tiếp, tỷ lệ aspect ratio có thể bị bóp méo nhẹ so với camera.
                # Đoạn code này vẽ trực tiếp theo tỷ lệ màn hình.
                lx = int(lm[i] * img.width())
                ly = int(lm[i+1] * img.height())
                
                # Vẽ điểm màu đỏ lên màn hình
                img.draw_circle(lx, ly, 3, image.COLOR_RED, -1)

        # In thông số FPS / Thời gian inference
        img.draw_string(10, 10, f"Infer: {t_infer:.1f}ms", image.COLOR_GREEN)

        # Hiển thị ra màn hình LCD
        disp.show(img)

if __name__ == "__main__":
    main()
