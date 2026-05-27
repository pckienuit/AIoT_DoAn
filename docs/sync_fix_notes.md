# Tài Liệu Sửa Lỗi Đồng Bộ Hóa Dữ Liệu (Sync Down) & Kẹt ID Chuyến Bay 999999

Tài liệu này ghi chép chi tiết nguyên nhân, giải pháp và kết quả sửa lỗi liên quan đến tính năng đồng bộ hóa (Sync Down) từ WebAdmin xuống thiết bị biên MaixCAM, cũng như các lỗi KeyError và sự cố kẹt/không xóa được cache trên thiết bị biên.

---

## 1. Mô tả sự cố

1. **Hiện tượng kẹt giao diện**: Khi thực hiện đồng bộ hóa danh sách hành khách xuống thiết bị biên MaixCAM từ giao diện quản trị WebAdmin, dữ liệu không được cập nhật đúng hoặc giao diện hiển thị trên thiết bị biên (LCD/MJPEG HUD) bị kẹt hiển thị thông tin của chuyến bay thử nghiệm `Flight 999999` cùng dữ liệu cũ, không chuyển sang thông tin chuyến bay thực tế vừa chọn.
2. **Lỗi KeyError: 'flight_number'**: Khi nhấn nút đồng bộ chuyến bay trên WebAdmin, hệ thống trả về lỗi `Error: 'flight_number'` (HTTP 500 Internal Server Error) trên API `/api/admin/sync/trigger` và không tạo được tệp tin cache trên MaixCAM.
3. **Lỗi không xóa được cache chuyến bay cũ (RAM Cache Leak)**: Khi nhấn nút "Xóa cache" cho một chuyến bay trên WebAdmin, tệp tin cache `.json` trên thẻ nhớ SD của thiết bị biên đã bị xóa thành công. Tuy nhiên, thiết bị biên vẫn tiếp tục nhận diện hành khách và hiển thị thông tin chuyến bay đã xóa trên HUD màn hình. Lệnh xóa cache chỉ có tác dụng một lần, các lần sau không thể giải phóng cache lưu ở RAM thiết bị biên.

---

## 2. Phân tích nguyên nhân gốc rễ

### 2.1 Lỗi không cập nhật tập tin trạng thái chuyến bay hoạt động (`active_flight.txt`)
- Ở chế độ đồng bộ cờ hiệu (**Flag-based sync**), hệ thống ghi đè ID chuyến bay mục tiêu vào `/root/active_flight.txt` trước khi tạo cờ hiệu `sync_now.flag`.
- Tuy nhiên, ở chế độ đồng bộ trực tiếp (**Direct SSH sync**), API server chỉ chạy lệnh Python trên thiết bị để thực hiện tác vụ tải cache (`CacheManager(cfg).sync(fid)`) mà **không cập nhật** tập tin `/root/active_flight.txt`.
- Do tập tin này không được cập nhật, vòng lặp chính của ứng dụng (`main.py` trên MaixCAM) vẫn đọc giá trị cũ (mặc định là `999999` sau khi chạy tính năng *Sync All*). Vì vậy, thiết bị tiếp tục đối sánh và hiển thị HUD của chuyến bay `999999`.

### 2.2 Lỗi truy vấn thiếu các cột JOIN khi tạo Face Payload (`KeyError: 'flight_number'`)
- Trong [admin_routes.py](file:///d:/AIoT_DoAn/server/admin_routes.py), để chuẩn bị payload đồng bộ lưu vào cache, server gọi hàm `build_face_payload(bk)` từ `server/face_routes.py`.
- Hàm `build_face_payload` yêu cầu đối tượng booking phải chứa đầy đủ thông tin chuyến bay như: `flight_number`, `flight_date`, `origin_city`, `dest_city`, `flight_status`.
- Tuy nhiên, câu truy vấn cũ của WebAdmin chỉ là:
  ```sql
  SELECT * FROM bookings WHERE qdrant_point_id = ? ...
  ```
  Bảng `bookings` trong cơ sở dữ liệu quan hệ chỉ lưu `flight_id` (khóa ngoại) mà không lưu trực tiếp số hiệu chuyến bay (`flight_number`) hay ngày bay. Việc thiếu các cột kết nối (JOIN) dẫn đến lỗi **KeyError: 'flight_number'** làm sập API sync.

### 2.3 Lỗi cơ chế Cache lưu ở RAM biên (RAM Cache Invalidation)
- Để tối ưu hóa hiệu năng, ứng dụng MaixCAM duy trì một RAM cache (`self._ram_cache`) chứa các khuôn mặt đã được giải mã của chuyến bay hoạt động để tránh việc liên tục đọc và giải mã file từ thẻ SD trên mỗi frame hình.
- Hàm `_read_cache` trong `sync_cache.py` được thiết kế như sau:
  ```python
  def _read_cache(self, flight_id: int) -> dict | None:
      if flight_id in self._ram_cache:
          return self._ram_cache[flight_id]
  ```
  Nếu ID chuyến bay đã có trên RAM cache, nó sẽ lập tức trả về dữ liệu trong RAM mà **không kiểm tra xem tệp tin trên đĩa SD còn tồn tại hay không**.
- Khi WebAdmin thực hiện lệnh xóa cache chuyến bay, tệp tin `/root/cache/flight_{id}.json` bị xóa khỏi đĩa SD của MaixCAM, nhưng RAM cache của ứng dụng `main.py` vẫn giữ nguyên bản sao dữ liệu cũ trong bộ nhớ.
- Đồng thời, WebAdmin API xóa cache không cập nhật lại `/root/active_flight.txt`, dẫn đến thiết bị biên vẫn coi chuyến bay đã xóa là chuyến bay hoạt động và tiếp tục lấy dữ liệu từ RAM cache để hiển thị.

---

## 3. Các chỉnh sửa đã thực hiện

### 3.1 Sửa đổi cơ chế thu hồi RAM Cache (RAM Cache Invalidation)
**Tập tin sửa đổi**: [MaixCAM_App/sync_cache.py](file:///d:/AIoT_DoAn/MaixCAM_App/sync_cache.py)

Cập nhật hàm `_read_cache` để kiểm tra sự tồn tại của tệp tin đĩa trước khi trả về dữ liệu RAM. Nếu tệp tin đĩa không tồn tại (đã bị xóa từ xa), tiến hành xóa khóa đó khỏi RAM cache và trả về `None`:

```python
    def _read_cache(self, flight_id: int) -> dict | None:
        path = self._cache_path(flight_id)
        if not os.path.exists(path):
            if flight_id in self._ram_cache:
                del self._ram_cache[flight_id]
            return None

        if flight_id in self._ram_cache:
            return self._ram_cache[flight_id]
```

### 3.2 Cập nhật WebAdmin API xóa cache
**Tập tin sửa đổi**: [server/admin_routes.py](file:///d:/AIoT_DoAn/server/admin_routes.py)

Trong hàm `_cleanup_flight_cache`, bổ sung logic kiểm tra xem chuyến bay vừa bị xóa cache có phải chuyến bay đang hoạt động (`active_flight.txt`) hay không. Nếu có, thực hiện xóa tệp tin `/root/active_flight.txt` để buộc thiết bị biên chuyển về trạng thái không có chuyến bay hoạt động hoặc chuyến bay mặc định:

```python
        # Check if this flight was the active flight. If so, clear it.
        try:
            rc, out = _ssh_run(client, "cat /root/active_flight.txt 2>/dev/null || echo ''")
            if rc == 0 and out.strip() == str(flight_id):
                _ssh_run(client, "rm -f /root/active_flight.txt")
        except Exception:
            pass
```

### 3.3 Sửa lỗi KeyError bằng truy vấn JOIN đầy đủ
**Tập tin sửa đổi**: [server/admin_routes.py](file:///d:/AIoT_DoAn/server/admin_routes.py)

Thay thế các câu truy vấn `SELECT * FROM bookings` đơn giản bằng truy vấn kết nối (JOIN) nhiều bảng để lấy đầy đủ thông tin cần thiết phục vụ cho `build_face_payload`.

### 3.4 Cập nhật WebAdmin API Server (Direct Mode Active Flight)
**Tập tin sửa đổi**: [server/admin_routes.py](file:///d:/AIoT_DoAn/server/admin_routes.py)

Cập nhật khối lệnh chế độ `direct` để ghi đè ID chuyến bay hoạt động lên thiết bị trước khi chạy lệnh Python sync.

### 3.5 Cập nhật CLI Sync Tool
**Tập tin sửa đổi**: [sync_maixcam.py](file:///d:/AIoT_DoAn/sync_maixcam.py)

Cập nhật hàm `run_direct_sync` trong CLI script để ghi đè tập tin active flight tương tự WebAdmin API.

### 3.6 Đồng bộ cấu hình IP mặc định của thiết bị biên
**Tập tin sửa đổi**: [MaixCAM_App/config.py](file:///d:/AIoT_DoAn/MaixCAM_App/config.py)

Cập nhật `server_url` mặc định về đúng dải IP mạng USB của máy chủ thực tế (`10.154.36.100:8000`).

---

## 4. Quy trình kiểm tra & Xác minh (Verification)

1. **Đồng bộ hóa chuyến bay**:
   Thực hiện đồng bộ hóa chuyến bay thực tế `#14331` từ WebAdmin.
   *Kết quả mong đợi*: Giao diện thiết bị biên hiển thị ID chuyến bay hoạt động là `14331` và có số lượng hành khách tương ứng.

2. **Xóa cache chuyến bay**:
   Nhấn nút "Xóa cache" cho chuyến bay `#14331` từ WebAdmin.
   *Kết quả mong đợi*:
   - Tệp `/root/cache/flight_14331.json` và `/root/active_flight.txt` bị xóa khỏi thiết bị biên.
   - Thiết bị biên lập tức nhận biết tập tin cấu hình bị xóa, dọn sạch RAM cache tương ứng.
   - Giao diện thiết bị biên tự động thu hồi/chuyển đổi HUD hiển thị, không còn hiển thị chuyến bay `#14331` hay nhận diện hành khách thuộc chuyến bay này nữa.
