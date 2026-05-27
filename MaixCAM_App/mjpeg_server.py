import socketserver
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

_latest_frame = None
_lock = threading.Lock()
_last_update_time = 0.0

# Load config settings for streaming
try:
    from config import load_config
    _CFG = load_config()
    _fps_limit = _CFG.get("stream_fps_limit", 15)
    _jpeg_quality = _CFG.get("stream_jpeg_quality", 70)
except Exception:
    _fps_limit = 15
    _jpeg_quality = 70

class MJPEGHandler(BaseHTTPRequestHandler):
    # Overwrite log_message to prevent flooding console output
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Connection', 'close')
            self.send_header('Max-Age', '0')
            self.send_header('Expires', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            # Send at constant FPS by repeating last frame — decouples stream
            # smoothness from AI pipeline speed (which may be only 5-7 FPS).
            stream_interval = 1.0 / max(1, _fps_limit)
            while True:
                with _lock:
                    frame = _latest_frame

                if frame is None:
                    time.sleep(0.01)
                    continue

                try:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(
                        ('Content-Length: %d\r\n\r\n' % len(frame)).encode('utf-8')
                    )
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
                except Exception:
                    break

                time.sleep(stream_interval)
        else:
            self.send_error(404)

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    # Each client connection runs in its own thread — no blocking between clients
    daemon_threads = True

def start_server(port=8080):
    server = ThreadedHTTPServer(('0.0.0.0', port), MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[mjpeg] Stream server started on port {port} (fps_limit={_fps_limit}, quality={_jpeg_quality})")
    return server

def configure(fps_limit, quality):
    global _fps_limit, _jpeg_quality
    _fps_limit = fps_limit
    _jpeg_quality = quality
    print(f"[mjpeg] Configured: fps_limit={fps_limit}, quality={quality}")

def update_frame(img):
    global _latest_frame, _last_update_time
    now = time.time()
    
    if _fps_limit > 0:
        min_interval = 1.0 / _fps_limit
        if now - _last_update_time < min_interval:
            return  # Throttle frame rate to save CPU
            
    try:
        _last_update_time = now

        jpeg_img = img.to_jpeg(_jpeg_quality)
        if jpeg_img:
            jpeg_bytes = jpeg_img.to_bytes()
            with _lock:
                _latest_frame = jpeg_bytes
    except Exception as e:
        print("[mjpeg] Encode error:", e)
