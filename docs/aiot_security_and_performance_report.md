# Báo Cáo Kiến Trúc, Bảo Mật Và Hiệu Năng Hệ Thống Check-in Bằng Khuôn Mặt (AIoT)

Tài liệu này tổng hợp toàn bộ các nội dung kỹ thuật cốt lõi đã được trao đổi liên quan đến kiến trúc hạ tầng, giao thức giao tiếp, các lớp mã hóa bảo mật, thiết kế cơ sở dữ liệu, và các giải pháp tối ưu hóa hiệu năng trên thiết bị biên nhúng của dự án.

---

## 1. Giao Thức Giao Tiếp Giữa Server Và Thiết Bị Biên (Edge Device)

Thiết bị biên (MaixCAM chạy vi xử lý RISC-V C906) kết nối và tương tác với FastAPI Server thông qua các tầng giao thức sau:
*   **Tầng Vật Lý & Liên Kết:** Thiết bị kết nối qua mạng **WiFi** cục bộ hoặc qua giao diện **USB Virtual Ethernet** với địa chỉ IP tĩnh mặc định là `10.154.35.1`.
*   **Tầng Ứng Dụng:** Sử dụng giao thức **HTTP/HTTPS (REST API)** để gửi/nhận dữ liệu.
*   **Các API giao tiếp chính (Triển khai trong lớp [CacheManager](file:///d:/AIoT_DoAn/MaixCAM_App/sync_cache.py#L153)):**
    1.  **Đồng bộ hóa dữ liệu (Sync Cache - GET):** Gửi yêu cầu tới `/api/sync/{flight_id}` để tải danh sách các vector khuôn mặt của hành khách trên chuyến bay về lưu cache dưới dạng tệp JSON trên thẻ nhớ MicroSD.
    2.  **Đối sánh Fallback (Fallback Match - POST):** Gửi yêu cầu tới `/api/face/match` kèm theo vector khuôn mặt đã mã hóa khi gặp tình huống Cache Miss (không đối sánh được dưới biên).

---

## 2. Hệ Thống Mã Hóa Kép (Dual-Cipher Security)

Hệ thống triển khai cơ chế bảo mật đầu cuối (End-to-End) thông qua việc kết hợp hai thuật toán mã hóa khác nhau ở các phân khu nhằm tối ưu giữa độ an toàn và hiệu năng phần cứng:

```text
  [Browser Client] ─── (AES-GCM-256) ───▶ [FastAPI Server] ─── (XTEA-CTR-128) ───▶ [Edge Device]
   (In-Transit)                            (Database RAM)                          (At-Rest Cache)
```

### 2.1 Mã hóa AES-GCM-256 (Web Client ➔ Server)
*   **Khái niệm AES đơn giản:** **AES (Advanced Encryption Standard)** là tiêu chuẩn mã hóa đối xứng an toàn nhất thế giới hiện nay, hoạt động bằng cách dùng một chiếc khóa bí mật duy nhất để xáo trộn dữ liệu thô thành dạng không thể đọc được và giải mã ngược lại.
*   **Tại sao dùng GCM (Galois/Counter Mode) thay vì chế độ thường (CBC)?** Chúng ta chọn AES-GCM vì nó thuộc nhóm **AEAD** (Mã hóa có xác thực) — vừa mã hóa bảo mật vừa sinh ra một **Thẻ xác thực (Authentication Tag)** để ngăn chặn hoàn toàn việc hacker can thiệp hoặc sửa đổi dữ liệu trên đường truyền.
*   **Cách hoạt động:** Trình duyệt web sử dụng **Web Crypto API** (mã hóa native tích hợp sẵn trong trình duyệt cho tốc độ tối đa và bảo mật vùng nhớ) để mã hóa vector 128D dạng thô thành bản mã Base64 kèm IV ngẫu nhiên trước khi truyền đi. Server giải mã thông qua hàm [decrypt_aes_gcm_vector](file:///d:/AIoT_DoAn/server/crypto_service.py#L27-L46).

### 2.2 Mã hóa XTEA-CTR-128 (Server ↔ Edge Device)
*   **XTEA là gì?** **XTEA (eXtended Tiny Encryption Algorithm)** là thuật toán mã hóa khối đối xứng siêu nhẹ, chỉ sử dụng các phép toán số học cơ bản (cộng, dịch bit, XOR) trên 64 vòng lặp. Nó lý tưởng cho các vi điều khiển và chip nhúng như RISC-V C906 của MaixCAM (vốn chỉ có 128MB RAM).
*   **Độ an toàn của luồng Sync:** Không hề kém an toàn vì:
    1.  Dữ liệu truyền tải vẫn được bao bọc bởi lớp bảo vệ thứ nhất là **HTTPS/TLS**.
    2.  Chế độ **CTR (Counter)** kết hợp số ngẫu nhiên dùng một lần (**nonce**) đảm bảo bản mã luôn thay đổi dù vector gốc giống nhau.
    3.  Bảo vệ dữ liệu tĩnh (**At-Rest Security**): Bản mã XTEA được ghi thẳng xuống thẻ nhớ SD. Thiết bị biên chỉ giải mã tạm thời vào bộ nhớ RAM khi chạy đối sánh, giảm thiểu nguy cơ lộ dữ liệu sinh trắc học nếu thẻ nhớ bị đánh cắp vật lý.

---

## 3. Quản Lý Các Loại Khóa Trong Hệ Thống

| Loại Khóa | Thuật toán | Vị trí lưu trữ | Vòng đời | Vai trò |
| :--- | :---: | :--- | :--- | :--- |
| **Session Key** | AES-GCM-256 | Trình duyệt Client | Ngắn hạn (mỗi phiên) | Mã hóa vector tại client trước khi truyền qua Internet. |
| **Master Key** | AES-GCM-256 | Server (`AES_SECRET_KEY`) | Dài hạn | Giải mã gói tin đăng ký từ Web Client để lưu vào Qdrant. |
| **Device Key** | XTEA-CTR-128 | Server & MaixCAM Edge | Dài hạn | Mã hóa dữ liệu đồng bộ cache và gói tin đối sánh fallback. |

### 3.1 Nghịch lý giải mã giữa hai khóa trong thực tế triển khai (Pre-shared Key)

Theo nguyên tắc toán học, nếu khóa mã hóa (Session Key) và khóa giải mã (Master Key) khác nhau hoàn toàn thì thuật toán đối xứng **AES-GCM** sẽ không thể giải mã được dữ liệu. Hệ thống hiện tại giải quyết vấn đề này như thế nào?

*   **Bản chất triển khai hiện tại:** Trong phiên bản prototype, cả **Web Client** và **Server** đang sử dụng cơ chế **Khóa đối xứng dùng chung (Pre-shared Key)**.
    *   Phía Client (trong tệp [app.js](file:///d:/AIoT_DoAn/web_stage3/app.js#L2)) và Server (trong tệp [crypto_service.py](file:///d:/AIoT_DoAn/server/crypto_service.py#L7-L10)) đều nạp chung một chuỗi dữ liệu nhị phân khóa: `94c8e763a8a3a31e2474db62c82e0fb58cc2a77ef7cb73f1d8c117b4abdc3d9d`.
    *   Do đó, dù tên gọi đại diện về mặt kiến trúc ở mỗi bên khác nhau (Session Key ở Web, Master Key ở Server), giá trị của chúng là **một chiếc chìa khóa duy nhất**, đảm bảo dữ liệu giải mã thành công.
*   **Triển khai thực tế cấp độ Production:** Để nâng tầm bảo mật khi triển khai thương mại, hệ thống sẽ được nâng cấp lên cơ chế **Key Exchange (như Diffie-Hellman / ECDH)** hoặc **Mã hóa bất đối xứng (Asymmetric Cryptography)** để trao đổi Session Key ngẫu nhiên của mỗi phiên đăng ký một cách an toàn mà không cần cấu hình tĩnh chìa khóa trên mã nguồn.

---

## 4. Thiết Kế Cơ Sở Dữ Liệu & Cơ Chế Liên Kết

Hệ thống sử dụng đồng thời hai loại cơ sở dữ liệu trên Server:

1.  **CSDL Quan hệ (SQLite cho Prototype / MySQL sẵn sàng cho Production):**
    *   Triển khai tại [database.py](file:///d:/AIoT_DoAn/server/database.py).
    *   Lưu trữ metadata có cấu trúc của hành khách, lịch trình, ghế ngồi và thông tin đặt vé (`bookings`).
2.  **CSDL Vector (Qdrant chạy qua Docker):**
    *   Triển khai tại [vector_service.py](file:///d:/AIoT_DoAn/server/vector_service.py).
    *   Lưu trữ vector embedding 128D của khuôn mặt để thực hiện tìm kiếm khoảng cách cosine nhanh bằng chỉ mục HNSW.

### Cơ chế kết nối giữa hai CSDL:
Chúng liên kết thông qua mô hình **Liên kết lỏng (Loosely Coupled)** bằng một cột khóa ngoại logic là **`qdrant_point_id`** (định dạng UUID) trong bảng `bookings` của SQL DB.
*   Khi khách hàng đăng ký khuôn mặt, vector được đẩy vào Qdrant sinh ra một UUID. UUID này được cập nhật vào trường `qdrant_point_id` trong SQL DB.
*   Khi đối sánh thành công, Qdrant trả về UUID. Server dùng UUID này truy vấn ngược lại SQL để tìm thông tin vé chi tiết và chuyển trạng thái đặt vé sang `checked_in`.

---

## 5. Các Phương Pháp Thử Nghiệm Và Chỉ Số Độ Tin Cậy (Benchmark)

Hệ thống đã được đo đạc hiệu năng và độ tin cậy thông qua 4 phương pháp chính:

*   **Python Offline CPU Benchmark:** Dùng `onnxruntime` đo độ trễ thô của mô hình ONNX trên CPU máy tính.
*   **Web Browser WASM Benchmark:** Dùng `onnxruntime-web` chạy qua WebAssembly trực tiếp trên trình duyệt web.
*   **Stress Test Server & Qdrant:** Giả lập tải đăng ký và đồng bộ đồng thời cho nhóm quy mô 10, 50, 100 và 200 hành khách qua script [run_maixcam_stress_test.py](file:///d:/AIoT_DoAn/scripts/run_maixcam_stress_test.py).
*   **Real-time Edge Profiling:** Đo FPS và độ trễ xử lý thực tế trực tiếp trên NPU của MaixCAM.

### Kết quả hiệu năng chi tiết:
*   **Mô hình V9 Landmarks:** Đạt độ trễ cực thấp **2.70 ms** trên Python CPU và **59.05 ms** trên WASM Browser.
*   **Mô hình ArcFace P3:** Đạt độ trễ **6.39 ms** trên Python CPU và **80.03 ms** trên WASM Browser (nhanh gấp đôi mô hình SFace cũ).
*   **Hiệu năng trên thiết bị biên MaixCAM:** YOLOv8n-Face chạy mất **11.17 ms**, tổng pipeline E2E mất **15.86 ms**, throughput toàn luồng đạt **53 FPS** nhờ tăng tốc NPU.
*   **Hiệu năng Server:** Đạt tỷ lệ đăng ký thành công **100%**, độ trễ API đăng ký trung bình **~285 ms**, tốc độ sync dữ liệu cho 200 khách hàng chỉ mất **~457 ms**.

### Các chỉ số độ tin cậy:
*   **Ngưỡng đối sánh trùng khớp nghiêm ngặt (Strict Threshold):** `Cosine Distance <= 0.045` (tương đương cosine similarity `score >= 0.955`).
*   **Tỷ lệ nhận diện sai (False Positive Rate):** **< 0.1%** (đạt tiêu chuẩn an toàn hàng không).
*   **Độ chính xác LFW:** Mục tiêu đạt **≥ 95%** (kiểm thử thực tế đạt score tương đồng lên tới **0.974**).
*   **Làm mịn tọa độ (EMA):** Sử dụng bộ lọc làm mịn Exponential Moving Average với hệ số $\alpha=0.35$ giúp triệt tiêu hiện tượng rung lắc camera (jittering), đảm bảo độ ổn định của vector trích xuất.

---

## 6. Kỹ Thuật Tối Ưu Hóa Tốc Độ Trên Thiết Bị Biên (Edge Device)

Để đạt tốc độ nhận diện mượt mà và tránh làm nghẽn bộ nhớ RAM 128MB của MaixCAM, các giải pháp tối ưu sau đã được áp dụng:

1.  **AI Frame Skipping (`ai_frame_interval: 3`):** Chỉ chạy YOLOv8n và V9 Landmarks mỗi 3 khung hình camera một lần. Các khung hình trung gian vẽ HUD dựa trên dữ liệu đệm cũ.
2.  **Phân tách chu kỳ nhận diện (`recognition_frame_interval: 9`):** Chỉ chạy ArcFace P3 trích xuất embedding mỗi 9 khung hình hoặc khi bộ đệm nhận diện hết hạn (1.5 giây).
3.  **Tách biệt luồng Stream và luồng AI:** Server stream video MJPEG chạy trên một luồng riêng giới hạn ở 15 FPS. Nếu AI chưa tính toán xong khung hình mới, stream server sẽ phát lại khung hình cũ, đảm bảo video giám sát không bị giật lag.
4.  **Chỉ đối sánh cục bộ trong RAM:** Không gọi API Server trực tiếp trong luồng camera để tránh đứng hình do trễ mạng (Network Timeout).
5.  **Giảm I/O thẻ nhớ:** Chỉ đọc các cờ hiệu cấu hình định kỳ tối đa 1 lần/giây thay vì kiểm tra mỗi frame.

---

## 7. Kỹ Thuật Lượng Tử Hóa Mô Hình (Quantization)

Mô hình trên thiết bị biên nhúng được phân tách kỹ thuật lượng tử hóa tùy thuộc vào đặc trưng tác vụ:
*   **Mô hình V9 Landmarks ➔ Lượng tử hóa INT8:** Tác vụ xác định tọa độ chấp nhận sai số nhỏ. INT8 giúp nén mô hình xuống **10.2 MB** và chạy siêu nhanh trên NPU.
*   **Mô hình ArcFace P3 ➔ Lượng tử hóa F16 (BF16/FP16):** Tác vụ trích xuất vector 128D yêu cầu độ chính xác toán học tuyệt đối để so khớp cosine. Lượng tử hóa INT8 sẽ làm biến dạng các góc vector gây sai số nhận dạng lớn. Vì vậy, hệ thống sử dụng định dạng F16 để cân bằng giữa độ chính xác và tốc độ tăng tốc của NPU.

---

## 8. WebAssembly (WASM) Trên Trình Duyệt

*   **Bản chất:** WASM cho phép trình duyệt chạy mã nguồn nhị phân biên dịch từ C++/Rust/Go trực tiếp trên CPU người dùng với tốc độ gần như ứng dụng native.
*   **Ứng dụng trong dự án:** Trình duyệt người dùng tự tải mô hình ONNX và sử dụng thư viện `onnxruntime-web` (chạy trên nền WASM) để thực thi pipeline phát hiện và trích xuất vector ngay tại chỗ. điều này giúp giảm tải CPU/băng thông cho Server và bảo vệ quyền riêng tư tuyệt đối cho khách hàng (không phải gửi ảnh thô lên mạng).

---

## 9. Sơ Đồ Pipeline Đơn Giản Hóa Của Toàn Hệ Thống

```text
   +-------------------+             +-----------------------+             +-------------------+
   |    WEB CLIENT     |             |    FASTAPI SERVER     |             |    EDGE DEVICE    |
   | (Browser Browser) |             |  (SQLite/MySQL + API) |             |  (MaixCAM RISC-V) |
   +-------------------+             +-----------------------+             +-------------------+
             │                                   │                                   │
      [Webcam Stream]                     +-------------+                      [Camera Loop]
             │                            |  Qdrant DB  |                            │
      [MediaPipe Face]                    +-------------+                    [YOLOv8n-Face INT8]
             │                                   │                                   │
      [V9 Landmarks]                             │                            [V9 Landmarks INT8]
             │                                   │                                   │
    [ArcFace P3 (FP32)]                          │                          [ArcFace P3 (F16)]
             │                                   │                                   │
     (AES-GCM Encrypt)                           │                                   │
             │                                   │                                   │
             └────────────── HTTPS ─────────────▶│                                   │
                     /api/face/register          │                                   │
                                                 │                                   │
                                                 ├───────────── HTTPS ──────────────▶│
                                                 │       /api/sync/{flight_id}       │
                                                 │          (XTEA-CTR Cache)         │
                                                 │                                   │
                                                 │                                   ▼
                                                 │                           [SD Card Encrypted]
                                                 │                                   │
                                                 │                                   ▼
                                                 │                           (RAM Decrypted)
                                                 │                                   │
                                                 │                              [Cosine Match]
                                                 │                                   │
                                                 │                              (Cache Miss)
                                                 │                                   │
                                                 │◀──────────── HTTPS ───────────────┘
                                                 │          /api/face/match          
```
