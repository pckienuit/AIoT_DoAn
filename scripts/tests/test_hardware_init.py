#!/usr/bin/env python3
"""
test_hardware_init.py — Tests model loading + camera + display initialization on MaixCAM.
"""
import paramiko
import sys
import time

HOST = "10.154.35.1"
USER = "root"
PASS = "root"

DIAG_SCRIPT = """
import sys
import gc
import time

print("[1] Importing modules...")
from maix import nn, camera, display, image
print("  Import successful!")

print("[2] Loading YOLOv8 face detector...")
detector = nn.YOLOv8(model="/root/models/yolov8n_face.mud", dual_buff=False)
print("  Detector loaded! Input size:", detector.input_width(), "x", detector.input_height())

print("[3] Loading V9 landmark model...")
lm = nn.NN("/root/models/face_detect_v9.mud")
print("  V9 landmark model loaded!")

print("[4] Loading ArcFace P3 recognition model...")
recog = nn.NN("/root/models/face_recognize_arcface_p3.mud")
print("  ArcFace P3 loaded!")

print("[5] Initializing Camera...")
try:
    cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
    print("  Camera initialized successfully!")
except Exception as e:
    print("  Camera initialization failed:", e)
    sys.exit(1)

print("[6] Initializing Display...")
try:
    disp = display.Display()
    print("  Display initialized successfully! Size:", disp.width(), "x", disp.height())
except Exception as e:
    print("  Display initialization failed:", e)
    sys.exit(1)

print("[7] Reading 10 frames from camera and showing on display...")
for i in range(10):
    img = cam.read()
    img.draw_string(10, 10, f"Frame {i}", scale=1.5, color=image.COLOR_GREEN)
    disp.show(img)
    print(f"  Frame {i} read and displayed")
    time.sleep(0.1)

print("[8] Done!")
"""

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, username=USER, password=PASS, timeout=10)
        print("Connected!")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    sftp = client.open_sftp()
    with sftp.open('/root/diag_hardware_init.py', 'w') as f:
        f.write(DIAG_SCRIPT)
    sftp.close()

    print("\nRunning hardware init diagnostics on device...")
    stdin, stdout, stderr = client.exec_command("killall -9 python3; killall -9 maixvision_server; python3 /root/diag_hardware_init.py")
    
    print("\n--- Output ---")
    print(stdout.read().decode())
    err = stderr.read().decode()
    if err:
        print("--- Stderr ---")
        print(err)

    client.exec_command("rm -f /root/diag_hardware_init.py")
    client.close()

if __name__ == "__main__":
    main()
