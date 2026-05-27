# Tài Liệu Sửa Lỗi Đồng Bộ Hóa Dữ Liệu (Sync Down) & Kẹt ID Chuyến Bay 999999

Tài liệu này ghi chép chi tiết nguyên nhân, giải pháp và kết quả sửa lỗi liên quan đến tính năng đồng bộ hóa (Sync Down) từ WebAdmin xuống thiết bị biên MaixCAM, cũng như sự cố thiết bị biên bị kẹt ở ID chuyến bay thử nghiệm `999999`.

---

## 1. Mô tả sự cố

- **Hiện tượng**: Khi thực hiện đồng bộ hóa danh sách hành khách xuống thiết bị biên MaixCAM từ giao diện quản trị WebAdmin, dữ liệu không được cập nhật đúng hoặc giao diện hiển thị trên thiết bị biên (LCD/MJPEG HUD) bị kẹt hiển thị thông tin của chuyến bay thử nghiệm `Flight 999999` cùng dữ liệu cũ, không chuyển sang thông tin chuyến bay thực tế vừa chọn.
- **Phạm vi ảnh hưởng**: Tính năng đồng bộ ở chế độ **Direct (SSH) Sync** trên cả WebAdmin và CLI script (`sync_maixcam.py`).

---

## 2. Phân tích nguyên nhân gốc rễ

### 2.1 Lỗi không cập nhật tập tin trạng thái chuyến bay hoạt động (`active_flight.txt`)
- Ở chế độ đồng bộ cờ hiệu (**Flag-based sync**), hệ thống ghi đè ID chuyến bay mục tiêu vào `/root/active_flight.txt` trước khi tạo cờ hiệu `sync_now.flag`.
- Tuy nhiên, ở chế độ đồng bộ trực tiếp (**Direct SSH sync**), API server chỉ chạy lệnh Python trên thiết bị để thực hiện tác vụ tải cache (`CacheManager(cfg).sync(fid)`) mà **không cập nhật** tập tin `/root/active_flight.txt`.
- Do tập tin này không được cập nhật, vòng lặp chính của ứng dụng (`main.py` trên MaixCAM) vẫn đọc giá trị cũ (mặc định là `999999` sau khi chạy tính năng *Sync All*). Vì vậy, thiết bị tiếp tục đối sánh và hiển thị HUD của chuyến bay `999999`.

### 2.2 Sự bất đồng bộ trong dải IP mạng (Subnet Discrepancy)
- Địa chỉ IP tĩnh kết nối card mạng USB ảo của thiết bị biên được cấu hình thực tế là `10.154.36.1` (phía máy chủ là `10.154.36.100`).
- Tuy nhiên, cấu hình mặc định trong mã nguồn [config.py](file:///d:/AIoT_DoAn/MaixCAM_App/config.py) của thiết bị biên lại đặt `server_url` trỏ về `http://10.154.35.1:8000`. Điều này gây ra lỗi timeout hoặc không thể kết nối tới server khi thiết bị biên thực hiện lệnh sync HTTP.

---

## 3. Các chỉnh sửa đã thực hiện

### 3.1 Cập nhật WebAdmin API Server
**Tập tin sửa đổi**: [server/admin_routes.py](file:///d:/AIoT_DoAn/server/admin_routes.py)

Trong endpoint `/api/admin/sync/trigger`, cập nhật khối lệnh chế độ `direct` để thực hiện ghi đè ID chuyến bay hoạt động lên thiết bị trước khi chạy lệnh Python sync:

```python
            if payload.mode == "direct":
                # Cập nhật active flight để thiết bị biên chuyển sang chuyến bay mới
                _ssh_run(client, f"echo {fid} > /root/active_flight.txt")
                rc, out = _ssh_run(
                    client,
                    f"cd /root && python -c \"from sync_cache import CacheManager; from config import load_config; cfg = load_config(); CacheManager(cfg).sync({fid})\" 2>&1 || true"
                )
                outputs.append(f"Direct sync output: {out}")
```

### 3.2 Cập nhật CLI Sync Tool
**Tập tin sửa đổi**: [sync_maixcam.py](file:///d:/AIoT_DoAn/sync_maixcam.py)

Tương tự WebAdmin API, cập nhật hàm `run_direct_sync` trong CLI script để ghi đè tập tin active flight khi người dùng chỉ định chuyến bay cụ thể qua tham số `--flight`:

```python
def run_direct_sync(client: paramiko.SSHClient, flight_id: int | None) -> None:
    if flight_id is None:
        # ... sync all flights ...
    else:
        # Cập nhật active flight lên thiết bị biên trước khi đồng bộ trực tiếp
        rc, out = ssh_run(client, "echo {} > /root/active_flight.txt".format(int(flight_id)))
        if rc != 0:
            raise RuntimeError("Failed to set active flight: {}".format(out))
        print("[direct] Active flight set to {}".format(flight_id))
        remote_script = r"""
        # ... logic sync cache ...
        """.replace("__FLIGHT_ID__", str(int(flight_id)))
```

### 3.3 Đồng bộ hóa cấu hình IP mặc định của thiết bị biên
**Tập tin sửa đổi**: [MaixCAM_App/config.py](file:///d:/AIoT_DoAn/MaixCAM_App/config.py)

Cập nhật `server_url` mặc định về đúng dải IP mạng USB của máy chủ thực tế:

```python
DEFAULTS = {
    "server_url": "http://10.154.36.100:8000",
    "flight_ids": [1],          # List of flight IDs to sync
    # ... các cấu hình khác ...
}
```

---

## 4. Quy trình kiểm tra & Xác minh (Verification)

Để kiểm tra lại tính đúng đắn sau sửa đổi, thực hiện các bước sau:

1. **Khởi động API Server**:
   Chạy máy chủ FastAPI bằng script có sẵn:
   ```bash
   ./start_server.bat
   ```
   Kiểm tra endpoint sức khỏe để chắc chắn DB đã sẵn sàng:
   `http://localhost:8000/health`

2. **Chạy CLI Sync**:
   Kích hoạt đồng bộ hóa cho chuyến bay thực tế (ví dụ chuyến bay `#14331` có dữ liệu khuôn mặt hợp lệ trong database):
   ```bash
   python sync_maixcam.py --flight 14331 --direct
   ```
   *Kết quả mong đợi*:
   - Xuất hiện thông báo: `[direct] Active flight set to 14331`.
   - Lệnh sync hoàn thành và in ra: `Synced 1 embeddings for flight 14331`.
   - File `/root/cache/flight_14331.json` được tạo thành công trên thiết bị biên.

3. **Kiểm tra trạng thái thiết bị**:
   Kết nối SSH tới MaixCAM (`10.154.36.1`) và kiểm tra nội dung file active flight:
   ```bash
   cat /root/active_flight.txt
   ```
   *Kết quả mong đợi*: In ra `14331` thay vì kẹt ở `999999`. Ứng dụng hiển thị HUD trên MJPEG Stream sẽ tự động tải cache mới và đối sánh chính xác khuôn mặt của hành khách thuộc chuyến bay `14331`.
