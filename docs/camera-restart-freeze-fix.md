# Bug Fix: Camera Stream Freezing on Restart

> **Ngày ghi nhận:** 2026-05-27  
> **Mức độ:** 🔴 Critical — phải rút nguồn mới khởi động lại được  
> **Trạng thái:** ✅ Fixed & Deployed

---

## 1. Mô Tả Lỗi

Khi chạy `restart_maixcam.py` để khởi động lại ứng dụng trên MaixCAM, luồng camera MJPEG bị kẹt hoàn toàn — không còn nhận frame mới. Ứng dụng mới khởi động nhưng camera không hoạt động. Cách duy nhất khắc phục là **rút nguồn và cắm lại thiết bị**.

```
[Triệu chứng]
$ python restart_maixcam.py
  → App killed, App restarted
  → http://10.154.36.1:8080 → frozen / no frames
  → Phải rút nguồn MaixCAM mới dùng được lại
```

---

## 2. Nguyên Nhân Gốc (Root Cause)

### 2.1 Cơ chế hoạt động của camera driver MaixCAM

MaixCAM sử dụng driver ISP/VENC cấp kernel (C++) để quản lý camera pipeline:

```
Camera Sensor (GC4653)
    ↓  MIPI CSI-2
ISP (Image Signal Processor)
    ↓
VENC (Video Encoder) — giữ DMA buffer lock
    ↓
Python maix.camera object
```

Khi `maix.camera.Camera()` được khởi tạo, driver C++ mở một **exclusive lock** trên buffer VENC/ISP. Lock này chỉ được giải phóng khi đối tượng bị destructor'd đúng cách qua `cam.close()`.

### 2.2 Vấn đề trong code cũ

Vòng lặp chính trong `main.py` (code cũ):

```python
# Code CŨ — KHÔNG có cleanup
def main():
    cam = camera.Camera(640, 480, ...)   # Mở lock VENC
    disp = display.Display()              # Mở framebuffer

    while not app.need_exit():           # Vòng lặp chính
        img = cam.read()
        ...
    
    # ← Hàm kết thúc tại đây, KHÔNG gọi cam.close() hay disp.close()
    # ← Python garbage collector có thể không chạy kịp trước khi process bị kill
```

Khi `restart_maixcam.py` gửi `SIGTERM`:

```
SIGTERM → Python process bị terminate ngay lập tức
         → cam.__del__() KHÔNG được gọi đáng tin cậy
         → VENC buffer lock KHÔNG được giải phóng
         → Tiến trình mới start → camera.Camera() → lỗi (lock còn đó)
         → Camera stream frozen
```

### 2.3 Tại sao `SIGTERM` không đủ?

`SIGTERM` mặc định trong Python không trigger cleanup thông qua `finally` blocks hoặc destructor objects. Quá trình terminate diễn ra ở C level, bỏ qua Python runtime cleanup.

---

## 3. Giải Pháp (Fix)

### 3.1 Signal Handler trong `main.py`

Thêm signal handler bắt `SIGTERM` và `SIGINT`, đặt cờ `_shutdown_requested` để main loop thoát sạch:

```python
# Thêm vào MaixCAM_App/main.py

import signal

_shutdown_requested = False

def _signal_handler(signum, frame):
    """Bắt SIGTERM/SIGINT → main loop thoát qua flag thay vì bị kill ngay."""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    print("[main] Received {} — requesting graceful shutdown...".format(sig_name))
    _shutdown_requested = True


def main():
    global _shutdown_requested

    # Cài signal handler TRƯỚC khi mở hardware
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    cam = camera.Camera(640, 480, ...)
    ...
```

### 3.2 `try/finally` bảo đảm cleanup luôn chạy

Bọc toàn bộ main loop trong `try/finally`. Khối `finally` **luôn được thực thi** dù loop thoát bình thường, do signal, hay do exception:

```python
    try:
        while not app.need_exit() and not _shutdown_requested:
            img = cam.read()
            # ... toàn bộ AI pipeline ...

    except Exception as e:
        print("[main] Fatal error in main loop: {}".format(e))
    finally:
        # CRITICAL CLEANUP — phải chạy trước khi process thoát
        print("[main] Shutting down — releasing hardware resources...")

        # 1. Đóng camera (giải phóng VENC/ISP buffer lock)
        try:
            cam.close()
            print("[main] Camera closed.")
        except Exception as e:
            print("[main] Camera close error: {}".format(e))

        # 2. Đóng display (giải phóng framebuffer)
        if disp:
            try:
                disp.close()
                print("[main] Display closed.")
            except Exception as e:
                print("[main] Display close error: {}".format(e))

        # 3. GC để giải phóng reference NN models
        gc.collect()
        print("[main] Cleanup complete. Safe to restart.")
```

> **⚠️ Lưu ý quan trọng:** Không dùng `del FACE_DET` bên trong `main()`.  
> Python scoping rule: nếu có bất kỳ assignment hoặc `del` nào cho một tên biến bên trong hàm, Python coi nó là **local variable** trong toàn bộ hàm đó — kể cả các dòng trước `del`. Điều này gây `UnboundLocalError` ngay tại `camera.Camera(..., FACE_DET.input_format())` ở đầu hàm.

### 3.3 Cập nhật `restart_maixcam.py`

Script restart được cập nhật để:
1. **Truncate log** trước khi kill (để detect cleanup message mới)
2. **Gửi SIGTERM** và chờ log xuất hiện dòng `"Cleanup complete. Safe to restart."`
3. **Chỉ force-kill** nếu cleanup không hoàn tất trong 8 giây

```python
CLEANUP_MARKER = "Cleanup complete. Safe to restart."
GRACEFUL_TIMEOUT = 8  # seconds

def wait_for_cleanup(timeout):
    """Poll log file cho đến khi thấy cleanup marker."""
    start = time.time()
    while time.time() - start < timeout:
        stdin, stdout, stderr = ssh.exec_command(
            f"grep -c '{CLEANUP_MARKER}' /root/main.log 2>/dev/null"
        )
        time.sleep(0.5)
        count = stdout.read().decode().strip()
        if count.isdigit() and int(count) > 0:
            print(">> Graceful cleanup confirmed in log.")
            return True
    return False

# Flow:
# 1. truncate log
# 2. killall -15 (SIGTERM) → trigger signal handler
# 3. wait_for_cleanup(8s) → chờ cam.close() / disp.close()
# 4. Nếu process còn sống → killall -9 + sleep 3s
# 5. Start app mới
```

---

## 4. Luồng Sau Khi Fix

```
restart_maixcam.py
    │
    ├─ truncate /root/main.log
    ├─ killall -15 python  ← SIGTERM
    │       ↓
    │   _signal_handler() chạy trong main.py
    │       ↓ _shutdown_requested = True
    │   while loop thoát sạch
    │       ↓
    │   finally block:
    │       cam.close()   ← VENC/ISP buffer được giải phóng ✓
    │       disp.close()  ← Framebuffer được giải phóng ✓
    │       gc.collect()
    │       print("Cleanup complete. Safe to restart.")
    │
    ├─ grep log → thấy "Cleanup complete" → xác nhận OK
    ├─ (process đã tự thoát)
    └─ nohup python -u /root/main.py > /root/main.log &
            ↓
        camera.Camera() thành công (lock đã free)
        Stream hoạt động bình thường ✓
```

---

## 5. Kết Quả Kiểm Tra

```
=== Sending SIGTERM to python processes ===
$ killall -15 python3 2>/dev/null; killall -15 python 2>/dev/null
Waiting up to 8s for graceful hardware release...
>> Graceful cleanup confirmed in log.

Processes exited cleanly after SIGTERM.

=== Starting app ===
$ nohup python -u /root/main.py > /root/main.log 2>&1 &

=== Process check after start ===
$ ps aux | grep main.py | grep -v grep
1042 root     python -u /root/main.py

=== Log output (first 20 lines) ===
[config] Loaded from /root/config.json
-- [I] load cvimodel from: /root/models/yolov8n_face.cvimodel
-- [I] load cvimodel from: /root/models/face_detect_v9.cvimodel
-- [I] load cvimodel from: /root/models/face_recognize_arcface_p3.cvimodel
[SAMPLE_COMM_SNS_ParseIni]-2185: Parse /mnt/data/sensor_cfg.ini
...camera initialized successfully...
```

**Trước fix:** Sau restart → camera stream frozen, phải rút nguồn.  
**Sau fix:** Sau restart → camera hoạt động ngay, không cần can thiệp thủ công.

---

## 6. Các File Thay Đổi

| File | Thay đổi |
|---|---|
| [`MaixCAM_App/main.py`](../MaixCAM_App/main.py) | Thêm `import signal`, `_signal_handler()`, `_shutdown_requested` flag, bọc main loop trong `try/finally` với `cam.close()` + `disp.close()` |
| [`restart_maixcam.py`](../restart_maixcam.py) | Thêm `wait_for_cleanup()`, truncate log trước kill, tăng wait logic |

---

## 7. Lưu Ý Khi Phát Triển

- **Lần restart đầu tiên** từ version cũ (chưa có signal handler) sẽ vẫn dùng force-kill và có thể cần thêm `sleep` sau khi kill. Từ lần thứ 2 trở đi sẽ luôn graceful.
- **Không bao giờ** dùng `del <module_global>` bên trong `main()` — xem lý do ở mục 3.2.
- Nếu thêm hardware resource mới (ví dụ audio, GPIO), nhớ thêm cleanup tương ứng vào khối `finally`.
- Log dòng `"[main] Cleanup complete. Safe to restart."` là **marker chính thức** xác nhận hardware đã được giải phóng an toàn.
