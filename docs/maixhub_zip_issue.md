# Vấn đề Cấu trúc thư mục khi nén file ZIP cho MaixHub / Hệ thống Linux

## 1. Bối cảnh sự cố
Khi nén mô hình ONNX và thư mục dữ liệu Calibration (tên là `images/`) để upload lên công cụ **MaixHub Converter** (hoặc các hệ thống Linux backend tương tự), hệ thống ném ra lỗi:
`maixhub_utils.exception.Param_Exception: images directory is required`

Ngay cả khi bên trong file ZIP đã chứa các ảnh như `images/calib_001.jpg`, hệ thống vẫn báo không tìm thấy thư mục `images`.

## 2. Bản chất của file ZIP (Implicit vs Explicit Directories)
Bên trong cấu trúc lõi của định dạng ZIP, **không có khái niệm thư mục thực sự**. Một file ZIP chỉ là một danh sách phẳng (flat list) chứa các bản ghi (entries).

### Cách 1: Nén kiểu ngầm định (Implicit)
Đây là cách mà Windows (`Compress-Archive`) hoặc WinRAR thường làm:
1. `face_detect_model.onnx`
2. `images/1.jpg`
3. `images/2.jpg`

Khi mở trên Windows, hệ điều hành tự phân tích đường dẫn `images/...` và vẽ ra một cái icon thư mục màu vàng. Tuy nhiên, **bản ghi gốc của thư mục `images` không hề tồn tại**. Các đoạn code kiểm tra thư mục đơn giản trên Linux (như `os.path.isdir()`) sẽ quét danh sách này và kết luận: *"Không có thư mục nào tên là images/ cả"*.

### Cách 2: Nén kiểu tường minh (Explicit)
Để các hệ thống khắt khe (như MaixHub) nhận diện được, danh sách trong ZIP phải có một bản ghi dành riêng cho thư mục:
1. `face_detect_model.onnx`
2. **`images/`** *(Bản ghi đặc biệt, dung lượng 0 byte, kết thúc bằng dấu `/`)*
3. `images/1.jpg`
4. `images/2.jpg`

Khi nhận được bản ghi số 2, trình kiểm tra của MaixHub sẽ cho qua bước validation.

## 3. Cách khắc phục bằng Python (`zipfile`)
Để tạo ra bản ghi tường minh (Explicit Directory Entry), ta sử dụng thư viện `zipfile` của Python và gọi hàm `writestr("folder_name/", "")` để chèn một bản ghi rỗng kết thúc bằng dấu gạch chéo `/`:

```python
import zipfile
import os

def create_zip():
    zip_name = "maixhub_upload_fixed.zip"
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Thêm file model
        zipf.write("face_detect_model_v3.onnx", arcname="face_detect_model_v3.onnx")
        
        # 2. Thêm bản ghi thư mục tường minh (BẮT BUỘC CHO MAIXHUB)
        zipf.writestr("images/", "")
        
        # 3. Thêm các file ảnh vào trong thư mục
        images_dir = "images"
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.endswith('.jpg'):
                    file_path = os.path.join(root, file)
                    arcname = os.path.join("images", file).replace("\\", "/")
                    zipf.write(file_path, arcname=arcname)
                    
    print(f"Created {zip_name} successfully!")

if __name__ == "__main__":
    create_zip()
```

## 4. Bài học rút ra (Takeaway)
- Khi deploy model từ môi trường Windows lên các server AI IoT hoặc Docker chạy Linux, không nên tin tưởng tuyệt đối vào công cụ nén mặc định của OS.
- Nếu server yêu cầu thư mục với tên cụ thể (như dataset, calibration, weights), hãy dùng kịch bản nén bằng Python (`zipfile` hoặc `shutil.make_archive`) để kiểm soát chính xác cấu trúc bên trong file ZIP.
