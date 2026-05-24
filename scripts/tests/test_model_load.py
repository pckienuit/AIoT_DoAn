#!/usr/bin/env python3
"""
test_model_load.py — Diagnoses model loading issues on MaixCAM.
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
print("[1] Importing maix modules...")
try:
    from maix import nn, camera, display
    print("  Import successful!")
except Exception as e:
    print("  Import failed:", e)
    sys.exit(1)

print("[2] Loading YOLOv8 face detector...")
try:
    detector = nn.YOLOv8(model="/root/models/yolov8n_face.mud", dual_buff=False)
    print("  Detector loaded! Input size:", detector.input_width(), "x", detector.input_height())
except Exception as e:
    print("  Detector load failed:", e)
    import traceback; traceback.print_exc()

print("[3] Loading V9 landmark model...")
try:
    lm = nn.NN("/root/models/face_detect_v9.mud")
    print("  V9 landmark model loaded!")
except Exception as e:
    print("  V9 landmark model load failed:", e)
    import traceback; traceback.print_exc()

print("[4] Loading ArcFace P3 recognition model...")
try:
    recog = nn.NN("/root/models/face_recognize_arcface_p3.mud")
    print("  ArcFace P3 loaded!")
except Exception as e:
    print("  ArcFace P3 load failed:", e)
    import traceback; traceback.print_exc()

print("[5] Memory status...")
import os
try:
    with open('/proc/meminfo', 'r') as f:
        print(f.read()[:200])
except Exception:
    pass
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
    with sftp.open('/root/diag_model_load.py', 'w') as f:
        f.write(DIAG_SCRIPT)
    sftp.close()

    print("\nRunning diagnostics on device...")
    stdin, stdout, stderr = client.exec_command("python3 /root/diag_model_load.py")
    
    print("\n--- Output ---")
    print(stdout.read().decode())
    err = stderr.read().decode()
    if err:
        print("--- Stderr ---")
        print(err)

    client.exec_command("rm -f /root/diag_model_load.py")
    client.close()

if __name__ == "__main__":
    main()
