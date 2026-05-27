# Kiến Trúc Cache và Đồng Bộ Dữ Liệu trên Thiết Bị Biên (Edge Device)

Tài liệu này mô tả chi tiết về quy trình đồng bộ, lưu trữ, quản lý bộ nhớ và đối sánh nhận diện khuôn mặt trên thiết bị biên MaixCAM trong hệ thống AIoT Tra cứu thông tin chuyến bay.

---

## 1. Luồng Đồng Bộ Dữ Liệu Xuống Thiết Bị Biên
Để đảm bảo thiết bị biên có thể hoạt động độc lập (offline) và đạt tốc độ nhận diện dưới 1ms, dữ liệu hành khách sẽ được đồng bộ (pre-sync) trước giờ cất cánh.

1. **Gọi API Đồng bộ:** Thiết bị biên gọi endpoint `GET /api/sync/{flight_id}`.
2. **Mã hóa (On-the-fly):** Server lấy dữ liệu vector (plaintext) từ RAM/Qdrant, mã hóa bằng thuật toán **XTEA-CTR (128-bit)** với khóa thiết bị (`device_secret_key`).
3. **Truyền tải:** Dữ liệu mã hóa được trả về thiết bị biên qua mạng (JSON format).
4. **Lưu trữ Cục bộ:** Thiết bị biên lưu trực tiếp JSON (chứa ciphertext, iv và thông tin vé) xuống **thẻ nhớ SD (SD Card)**. Dữ liệu *không* được giải mã ở bước này.

---

## 2. Quy Trình Đối Sánh (Matching Pipeline)
Khi hành khách đứng trước camera, quy trình tìm kiếm vector khuôn mặt được thực hiện qua cấu trúc phân tầng (Tiered Matching) để tối ưu cả về tốc độ và bảo mật:

1. **So sánh trên RAM (L1 Cache - Nhanh nhất):**
   - Hệ thống kiểm tra xem dữ liệu chuyến bay đã có sẵn trên RAM chưa (`self._ram_cache`).
   - Nếu có, thuật toán Cosine Similarity chạy trực tiếp trên các mảng Float32. Thời gian hoàn thành: `< 1ms`.
2. **So sánh trên Thẻ nhớ SD (L2 Cache):**
   - Nếu dữ liệu chưa có trên RAM, hệ thống đọc file `flight_{id}.json` từ thẻ nhớ SD.
   - Thực hiện giải mã XTEA-CTR các vector khuôn mặt, chuẩn hóa (L2 Normalize) và đẩy vào RAM để thực hiện so sánh.
3. **So sánh trên Cơ sở dữ liệu Cloud (L3 Fallback):**
   - Nếu không tìm thấy bất kỳ sự trùng khớp nào trong Cache cục bộ, thiết bị biên gửi vector khuôn mặt mới trích xuất (được mã hóa) lên Server (API `/api/face/match`) để đối sánh trên toàn bộ CSDL Vector (Qdrant).

---

## 3. Quản Lý Dung Lượng & Bảo Mật

### Bảo Mật Dữ Liệu Sinh Trắc Học
Nếu dữ liệu khuôn mặt (Vector 128D) lưu dạng Plaintext trên thẻ SD, kẻ gian có thể đánh cắp thẻ và tái tạo lại định danh hành khách. Do đó:
- **Tất cả dữ liệu lưu trữ tĩnh (SD Card) đều bị mã hóa**.
- Hệ thống chỉ giải mã và lưu giữ Plaintext tạm thời **trong vùng nhớ RAM**. Khi thiết bị mất điện, toàn bộ dữ liệu sinh trắc học Plaintext sẽ tự động bốc hơi, đảm bảo an toàn vật lý tuyệt đối.

### Ước Tính Dung Lượng Thẻ Nhớ (SD Card)
- 1 Chuyến bay (200 hành khách) chiếm khoảng **~226 KB** không gian trên thẻ SD.
- Với thẻ nhớ tiêu chuẩn 8GB, thiết bị biên có sức chứa vật lý lên đến hàng chục nghìn chuyến bay, gần như không có giới hạn về storage cục bộ.

---

## 4. Giới Hạn Phần Cứng & Khả Năng Lưu Trữ trên RAM

MaixCAM sử dụng chip RISC-V với dung lượng RAM giới hạn là **128MB**. Mặc dù thẻ nhớ cực kỳ rộng rãi, nhưng giới hạn chịu tải thực sự nằm ở bộ nhớ RAM.

**Phân bổ RAM hiện tại:**
- Linux OS & System Drivers: ~50 - 60 MB
- AI Models (YOLO Face, v9 Landmarks, ArcFace P3): ~30 - 35 MB
- Camera Buffers & MJPEG Stream: ~10 - 15 MB
- Python Interpreter & Network: ~10 - 15 MB
- **👉 RAM trống khả dụng cho Cache: ~20 MB - 25 MB**

**Giới Hạn Chuyến Bay Trên RAM (An Toàn):**
- Overhead của Python (dict, con trỏ object list) khiến 1 chuyến bay (200 hành khách) chiếm **~1.0 MB - 1.2 MB** RAM.
- Để tránh bị Linux OOM Killer làm crash ứng dụng (cần khoảng 5-8MB safety buffer), thiết bị biên chỉ có thể **nạp tối đa khoảng 10 đến 15 chuyến bay cùng lúc** lên RAM.
- Điều này tương đương với năng lực phục vụ đồng thời **2,000 - 3,000 hành khách** (Cache Hit).

---

## 5. Chiến Lược Mở Rộng Cho Sân Bay Lưu Lượng Lớn (Vd: Tân Sơn Nhất)

Tại sảnh chính (Main Hall) của sân bay quốc tế như Tân Sơn Nhất, tần suất có thể đạt mức **44 - 48 chuyến bay mỗi giờ**. Với giới hạn an toàn 10-15 chuyến bay trên RAM, một thiết bị biên độc lập sẽ gặp khó khăn. Các chiến lược giải quyết bao gồm:

1. **Di dời thiết bị tới Cửa Khởi Hành (Boarding Gates):**
   - Thay vì lắp đặt ở sảnh chính, lắp MaixCAM tại từng cổng ra máy bay.
   - Tại mỗi cổng, chỉ cần Cache 1-2 chuyến bay chuẩn bị khởi hành. Khả năng Cache Hit đạt >99% với mức sử dụng RAM chỉ ~2MB.
2. **Triển khai Thuật toán LRU Cache trên RAM (Nếu giữ ở Main Hall):**
   - Thiết bị tải về thẻ SD toàn bộ 48 chuyến bay (chỉ tốn ~10MB SD Card).
   - Thiết lập thuật toán LRU (Least Recently Used) để chỉ nạp tối đa 10 chuyến bay (có lịch bay sát giờ nhất) lên RAM. Các chuyến bay bị đẩy ra (Evict) sẽ được giải phóng khỏi RAM để nhường chỗ, khi cần thiết sẽ được giải mã lại từ thẻ SD.
3. **Mô hình 3-Tier (Edge - Fog - Cloud):**
   - Nếu giữ nguyên thiết bị ở Sảnh chính, thay vì để thiết bị biên tự ôm đồm, lắp thêm một Local Server (Fog Node) tại sân bay chạy Qdrant.
   - MaixCAM chỉ chịu trách nhiệm trích xuất vector khuôn mặt và gửi request nội bộ tới Local Server qua WiFi LAN, duy trì tốc độ < 10ms mà hoàn toàn không lo giới hạn RAM.
