# Plan — Nâng Cấp Giao Diện Camera Edge (MaixCAM HUD Overlay)

Dự án AIoT Flight có giao diện web được thiết kế theo phong cách tối giản công nghiệp (dark industrial/airport terminal) rất hiện đại với tông màu đen làm chủ đạo, kết hợp lưới nền (grid texture) và các màu nhấn nổi bật (Neon Orange, Neon Lime/Green). Tuy nhiên, giao diện hiển thị camera và HUD trên thiết bị Edge (MaixCAM) hiện tại vẫn còn đơn điệu, chỉ bao gồm các đường vẽ chữ nhật liền màu xanh/đỏ đơn giản và các chuỗi text raw trắng không có cấu trúc.

Tài liệu này lập kế hoạch chi tiết để tái thiết kế giao diện HUD Camera trên Edge nhằm đồng bộ hoàn toàn với ngôn ngữ thiết kế cao cấp của Web Client.

---

## 🎨 Định Hướng Thiết Kế HUD Camera Mới

### 1. Khung Định Vị Bán Thể (Holographic target box)
*   **Hiện tại**: Vẽ hình chữ nhật liền nét quanh khuôn mặt (`COLOR_GREEN` hoặc `COLOR_RED`).
*   **Cải tiến**: Thay thế bằng **4 góc ngắm dạng bracket (`[ ]`)** tạo hiệu ứng khóa mục tiêu (target acquisition) hiện đại của camera AI.
*   **Hiệu ứng quét (Scanline)**: Khi đang trong trạng thái định vị/nhận diện, vẽ một đường quét laser (scanline) chạy quét dọc lên/xuống bên trong khung mặt bằng cách sử dụng hàm lượng giác `sin` theo thời gian.

### 2. Mạng Lưới Nhận Diện Khuôn Mặt (Face Mesh Wireframe)
*   **Hiện tại**: Vẽ 5 điểm mốc (landmarks) tròn với 5 màu sắc ngẫu nhiên.
*   **Cải tiến**:
    *   Đồng bộ màu sắc 5 điểm mốc về một màu Neon duy nhất (Neon Lime hoặc Neon Cyan) để tạo cảm giác đồng nhất.
    *   Vẽ thêm các đường liên kết mờ nối các điểm mốc (mắt-mắt, mắt-mũi, mũi-khóe miệng) tạo thành một khung mạng lưới (mesh wireframe) mô phỏng cấu trúc khuôn mặt đang được phân tích toán học.

### 3. Thẻ Thông Tin Nhận Diện (Premium Boarding Pass Panel)
*   **Hiện tại**: Vẽ một hình chữ nhật rỗng màu xanh lá ở góc trái màn hình chứa các dòng text rời rạc.
*   **Cải tiến**: Thiết kế lại toàn bộ Panel này thành một **Thẻ lên tàu ảo (Virtual Boarding Pass)**:
    *   Vẽ một khung viền đôi tinh tế (Double-line border) hoặc chia bố cục thành các cột rõ ràng.
    *   Bảng màu đồng bộ: Tên hành khách (UPPERCASE, màu trắng sáng), Cổng ra & Số ghế (màu Neon Orange nổi bật), Trạng thái chuyến bay (màu Neon Lime/Green).
    *   Sử dụng các ký tự ASCII định dạng gọn gàng để giả lập các icon/divider (`> `, `|`, `[ ]`).

### 4. Thanh HUD Hệ Thống ở Cạnh Dưới (System Status HUD)
*   **Hiện tại**: Dòng chữ thô: `v9+P3 | cache:10 | flight:999999 | th:0.045`.
*   **Cải tiến**: Thiết kế thành một thanh trạng thái hệ thống (System Status Bar):
    *   Chia làm 3 phân vùng rõ ràng:
        *   **Left**: Trạng thái thiết bị dạng LED: `● SYS: OK` hoặc `● CACHE: 12 PAX` (màu xanh Neon).
        *   **Center**: Mã chuyến bay đang hoạt động kèm trạng thái: `ACTIVE FLIGHT: VN123 [BOARDING]`.
        *   **Right**: Tham số AI: `THRES: 0.045 | FPS: 15`.

---

## 🛠️ Đề Xuất Thay Đổi

### Component: MaixCAM App (Edge)

#### [MODIFY] [display.py](file:///d:/AIoT_DoAn/MaixCAM_App/display.py)
*   Cập nhật và định nghĩa lại bảng màu trùng khớp mã Hex của web (chuyển sang hệ RGB 888 cho MaixCAM):
    *   `COLOR_PRIMARY` (Neon Orange): `Color(255, 107, 26)`
    *   `COLOR_ACCENT` (Neon Lime): `Color(214, 255, 63)`
    *   `COLOR_OK` (Neon Green): `Color(86, 240, 138)`
    *   `COLOR_DANGER` (Neon Red): `Color(255, 79, 79)`
    *   `COLOR_TEXT_MAIN` (Cream White): `Color(245, 241, 223)`
    *   `COLOR_TEXT_MUTED` (Warm Gray): `Color(169, 162, 139)`
*   Viết hàm tiện ích phụ `_draw_target_brackets(img, x, y, w, h, color, bracket_len=15, thickness=2)` để vẽ 4 góc ngắm thay thế cho `draw_rect`.
*   Viết hàm `_draw_scanline(img, x, y, w, h, color, now)` để tạo tia laser quét động.
*   Cải tiến hàm `draw_match_result` để thiết kế lại bảng thông tin Virtual Boarding Pass 2 cột sắc nét.
*   Cải tiến hàm `draw_hud` để hiển thị thanh hệ thống cân đối, có cấu trúc.
*   Cải tiến hàm `draw_no_match` để hiển thị cảnh báo đẹp mắt hơn với khung đỏ rỗng và text cảnh báo cảnh giác.

#### [MODIFY] [main.py](file:///d:/AIoT_DoAn/MaixCAM_App/main.py)
*   Chuyển đổi màu sắc của landmark trong `LM_COLORS` thành đồng màu `COLOR_ACCENT` hoặc `COLOR_CYAN`.
*   Thực hiện vẽ các đường nối (lines) giữa các điểm landmark để tạo hiệu ứng mạng lưới Face Mesh.
*   Bổ sung tham số `now` (thời gian thực) và logic vẽ scanline chạy liên tục bên trong khung nhận diện mặt khi đang quét/chưa khớp.
*   Điều chỉnh các lệnh vẽ hộp ranh giới (bounding box) gốc sang sử dụng cấu trúc vẽ góc ngắm và đường quét mới.

---

## 📋 Danh Sách Nhiệm Vụ (Task Breakdown)

### Phase 1: Nền tảng & Cập nhật display.py
*   **Task 1.1**: Định nghĩa lại bảng màu và các hằng số màu sắc đồng bộ với file CSS của web.
    *   **Input**: Các giá trị CSS biến màu từ `base.css`.
    *   **Output**: Các hằng số `COLOR_*` được cập nhật trong `display.py`.
    *   **Verify**: Import thành công các màu sắc mà không gây lỗi cú pháp.
*   **Task 1.2**: Xây dựng cơ chế vẽ góc ngắm ngắm (brackets) và tia quét laser động.
    *   **Input**: Tọa độ `x, y, w, h` từ bộ phát hiện khuôn mặt.
    *   **Output**: Hàm vẽ góc ngắm và hàm tia quét sử dụng hàm `sin` dựa trên `time.time()`.
    *   **Verify**: Chạy thử hàm vẽ trên một ảnh giả lập cho kết quả 4 góc vuông bo đều góc và tia laser di chuyển mượt mà.
*   **Task 1.3**: Tái cấu trúc Panel Boarding Pass và HUD hệ thống.
    *   **Input**: Dữ liệu chuyến bay từ `CacheManager`.
    *   **Output**: Panel Boarding Pass 2 cột cân đối, thanh HUD chân trang định dạng đẹp mắt.
    *   **Verify**: Code biên dịch không lỗi và giao diện layout hiển thị đúng vị trí (không bị tràn viền 640x480).

### Phase 2: Tích hợp logic vẽ vào pipeline chính (main.py)
*   **Task 2.1**: Tích hợp Face Mesh Wireframe.
    *   **Input**: Tọa độ 5 điểm landmark `lm_abs`.
    *   **Output**: Vẽ các đoạn thẳng nối 5 điểm mốc (LE-RE, LE-N, RE-N, N-LM, N-RM, LM-RM).
    *   **Verify**: Các đường nối tạo thành hình đa giác khuôn mặt cân đối trên khuôn mặt người.
*   **Task 2.2**: Thay thế hoàn toàn các lệnh vẽ rect/đường kẻ cũ sang giao diện HUD mới.
    *   **Input**: Pipeline phát hiện và đối sánh mặt trong `main.py`.
    *   **Output**: Tích hợp các hàm vẽ cải tiến động từ `display.py` vào vòng lặp camera.
    *   **Verify**: Chạy chương trình, kiểm tra stream MJPEG hiển thị đúng giao diện thiết kế mới.

---

## 🔬 Kế Hoạch Xác Minh (Verification Plan)

### Kiểm thử trên máy tính (Local Emulation)
*   Chạy script giả lập `scripts/tests/benchmark_edge_pc.py` (nếu có) hoặc viết một script test nhỏ tải ảnh tĩnh, gọi các hàm vẽ trong `display.py` và lưu lại thành file `.jpg` để đánh giá cảm quan trực quan.
*   Kiểm tra tính tương thích của thư viện `maix` khi vẽ.

### Kiểm thử trên thiết bị thật (Physical Edge Verification)
1.  Upload code đã cập nhật lên thiết bị MaixCAM qua lệnh SSH/SFTP (sử dụng script `deploy_to_device.py` hoặc chạy thủ công).
2.  Khởi chạy dịch vụ `main.py` trên MaixCAM:
    ```bash
    cmd /c "python deploy_to_device.py"
    ```
3.  Truy cập vào luồng livestream MJPEG tại địa chỉ `http://10.154.36.1:8080` (hoặc IP thiết bị tương ứng).
4.  Tiến hành đưa khuôn mặt trước camera để kiểm tra:
    *   **Trạng thái chưa nhận diện**: Khung mục tiêu có dạng góc ngắm `[ ]` màu cam kèm tia quét chạy dọc, các điểm mốc kết nối thành lưới face mesh màu lime.
    *   **Trạng thái khớp thành công**: Khung mục tiêu chuyển sang màu xanh lá, bảng Boarding Pass hiển thị đầy đủ thông tin hành khách phân cột rõ ràng, HUD chân trang hiển thị trạng thái khớp chuẩn.
    *   **Trạng thái không khớp**: Khung chuyển sang màu đỏ ranh giới kèm dòng chữ cảnh báo "NO BOOKING FOUND".
