# Báo cáo Tối ưu hóa Webcam Face Tracking Pipeline (v2)

Tài liệu này ghi lại toàn bộ các chỉnh sửa và tối ưu hóa đã được thực hiện để chuyển đổi từ hệ thống nhận diện khuôn mặt phụ thuộc vào Haar Cascade sang một hệ thống nhận diện và tracking hoàn toàn dựa trên Deep Learning (sử dụng mô hình MobileNetV2 fine-tuned).

## 1. Loại bỏ hoàn toàn Haar Cascade (Model-only Pipeline)
- **Vấn đề cũ:** Cần dùng `haarcascade_frontalface_default.xml` của OpenCV để tìm khuôn mặt trước khi đưa vào mô hình dự đoán Landmark. Haar Cascade chạy trên CPU, độ nhạy kém với ánh sáng yếu và bị giới hạn bởi các góc nghiêng.
- **Giải pháp:** Sử dụng chính mô hình Deep Learning để làm cả 2 nhiệm vụ: Phát hiện (Pass 1) và Căn chỉnh chi tiết (Pass 2). Điều này đảm bảo tính nhất quán tuyệt đối về luồng dữ liệu (domain consistency) từ lúc train cho đến lúc inference.

## 2. Tìm kiếm dạng lưới xử lý hàng loạt (Batched Grid-Search)
- **Vấn đề cũ:** Việc quét tìm khuôn mặt ban đầu chỉ tập trung ở 3 vùng chính giữa màn hình, khiến model bị "mù" nếu người dùng đứng ở sát mép trái/phải màn hình.
- **Giải pháp:** Xây dựng thuật toán Sliding Window với 4 kích cỡ (100%, 80%, 60%, 40% màn hình). Tất cả các ô cắt (crops) được gom thành một Batch duy nhất và đẩy vào GPU. GPU xử lý đồng loạt 30-40 ô cắt chỉ trong vài mili-giây, tìm ra ô có điểm số `face score` cao nhất.

## 3. Theo dõi chủ động (Active Tracking) & Chống giật (Anti-Jitter)
- **Vấn đề cũ:** Nếu dùng Pass 1 (quét toàn màn hình) liên tục sẽ gây nặng máy. Nếu dùng ngắt quãng (mỗi 3 frame 1 lần), khung hình sẽ bị khựng lại (staircase effect) khiến Landmarks bị nảy (rubber-band jitter).
- **Giải pháp:** Xây dựng vòng lặp **Active Tracking**.
  - Pass 1 chỉ chạy đúng 1 lần khi không có khuôn mặt (Scanning).
  - Khi đã bắt được mặt, hệ thống lấy chính kết quả vị trí của frame hiện tại để tạo khung cắt (Crop Box) cho frame tiếp theo.
  - Kết quả: Vòng lặp phản hồi (Feedback loop) liên tục giúp khung hình bám theo mặt mượt mà từng pixel trên mỗi frame mà không cần tốn tài nguyên chạy lại Pass 1.

## 4. Chống trôi khung hình (Divergence / Drift Prevention)
- **Vấn đề cũ:** Nếu lấy Bounding Box dự đoán của model để cập nhật độ rộng/cao của khung cắt ở frame tiếp theo, một sai số nhỏ (VD: model đoán rộng hơn 2%) sẽ bị cộng dồn theo cấp số nhân, khiến khung cắt to ra vô hạn hoặc trôi tuột khỏi màn hình. Việc làm mượt (EMA) trên hệ tọa độ ảo (crop coords) cũng gây ra hiện tượng cộng hưởng sai số này.
- **Giải pháp:** 
  1. Đổi toàn bộ tọa độ về **Tọa độ tuyệt đối (Absolute Frame Coords)** trước khi áp dụng thuật toán làm mượt (EMA).
  2. **Neo kích thước khung cắt vào Landmarks:** Kích thước của khung theo dõi (Tracking Box) được tính toán nghiêm ngặt dựa trên khoảng cách vật lý giữa 2 mắt và miệng (`core_w`, `core_h`). Vì khoảng cách này không thể tự phình to, khung Tracking Box bị khóa chặt và vô cùng ổn định.

## 5. Loại bỏ Nhiễu sáng (False-Positive Rejection)
- **Vấn đề cũ:** Dataset CelebA không có dữ liệu nhiễu là "bóng đèn huỳnh quang sáng rực". Do đó, mô hình CNN dễ bị đánh lừa và chấm điểm `0.99` cho các bóng đèn trần.
- **Giải pháp:** Áp dụng ràng buộc hình học (Geometric Sanity Check). Nếu mô hình đang bám vào một bóng đèn, các điểm Landmarks (mắt, mũi, miệng) sẽ bị nén lại thành một điểm rất nhỏ ở giữa nguồn sáng. Mã nguồn sẽ tự động kiểm tra: nếu khoảng cách giữa 2 mắt `core_w < 15` pixels, hệ thống sẽ ngay lập tức hủy Tracking và quét lại từ đầu, bỏ qua bóng đèn.

## 6. Sửa lỗi nội suy viền ảnh (Close-up Border Replicate)
- **Vấn đề cũ:** Khi người dùng dí sát mặt vào camera, phần trán hoặc cằm bị tràn ra khỏi màn hình. Ban đầu, phần tràn này được bù bằng các pixel màu đen (`zeros`), tạo ra các đường viền sắc nét kì lạ khiến mô hình bị nhầm lẫn và nhận diện sai Landmark.
- **Giải pháp:** Sao chép nguyên vẹn thuật toán cắt ảnh từ file `train_v8.py`. Sử dụng `cv2.BORDER_REPLICATE` (kéo dãn các pixel nằm ở mép màn hình). Mô hình đã từng được train với kiểu ảnh bị kéo dãn này nên nó lập tức hiểu và dự đoán Landmark chính xác dù mặt bị khuất khỏi màn hình.

## 7. Cải thiện Hiển thị Bounding Box
- **Thay đổi:** Khung CelebA Bounding Box nguyên bản chỉ bao quanh mắt và mũi (gây cảm giác chưa đúng cho người dùng). File `webcam_test_v2.py` hiện đã tự động dùng khoảng cách từ Landmarks để nội suy ra một Bounding Box chuẩn xác bao phủ toàn bộ vùng đầu (từ trán tới cằm), giống hệt như trải nghiệm sử dụng Haar Cascade.
