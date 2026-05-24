import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

_latest_frame = None
_lock = threading.Lock()

class MJPEGHandler(BaseHTTPRequestHandler):
    # Overwrite log_message to prevent flooding console output
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        global _latest_frame
        if self.path == '/':
            self.send_response(200)
            self.send_header('Connection', 'close')
            self.send_header('Max-Age', '0')
            self.send_header('Expires', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            last_frame = None
            while True:
                with _lock:
                    frame = _latest_frame
                
                if frame is None or frame is last_frame:
                    time.sleep(0.01)
                    continue
                
                try:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(frame)}\r\n\r\n'.encode('utf-8'))
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
                    last_frame = frame
                except Exception:
                    break
        else:
            self.send_error(404)

class ThreadedHTTPServer(HTTPServer):
    # Use standard HTTP server but ensure we don't block
    pass

def start_server(port=8080):
    server = ThreadedHTTPServer(('0.0.0.0', port), MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[mjpeg] Stream server started on port {port}")
    return server

def update_frame(img):
    global _latest_frame
    try:
        jpeg_img = img.to_jpeg(80)
        if jpeg_img:
            jpeg_bytes = jpeg_img.to_bytes()
            with _lock:
                _latest_frame = jpeg_bytes
    except Exception as e:
        print("[mjpeg] Encode error:", e)
