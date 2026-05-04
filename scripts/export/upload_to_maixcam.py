"""
Upload face detection + landmark models len MaixCAM qua SSH/SFTP.

Yeu cau: pip install paramiko

Model YOLOv8 face: https://wiki.sipeed.com/maixpy/doc/zh/vision/face_detection.html
- yolov8n_face.mud (nhanh, nhe) - cho MaixCAM
- yolo11s_face.mud - cho MaixCAM2
"""

import paramiko
import os
import sys

MAIXCAM_HOST = "10.154.36.1"
MAIXCAM_PORT = 22
MAIXCAM_USER = "root"
MAIXCAM_PASS = "root"

LOCAL_DIR  = os.path.join("models", "exports")
REMOTE_DIR = "/root/models"

MODELS = [
    ("yolov8n_face.mud", "Face detector (YOLOv8)"),
    ("yolov8n_face_320.cvimodel", "Face detector cvimodel"),
    ("face_detect_v3.mud", "Landmark model (cua ban)"),
]

MAIN_SCRIPT = os.path.join("MaixCAM_App", "main.py")


def main():
    print(f"Ket noi SSH toi MaixCAM ({MAIXCAM_HOST})...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(MAIXCAM_HOST, port=MAIXCAM_PORT,
                       username=MAIXCAM_USER, password=MAIXCAM_PASS, timeout=10)
    except Exception as e:
        print(f"[LOI] Khong the ket noi: {e}")
        return

    print("Ket noi thanh cong!")
    sftp = client.open_sftp()
    client.exec_command(f"mkdir -p {REMOTE_DIR}")
    
    success_count = 0
    for model_file, desc in MODELS:
        local_path  = os.path.join(LOCAL_DIR, model_file)
        remote_path = f"{REMOTE_DIR}/{model_file}"

        if not os.path.exists(local_path):
            print(f"[BO QUA] File khong ton tai: {local_path}")
            continue

        print(f" Dang upload [{desc}]...")
        try:
            sftp.put(local_path, remote_path)
            print(f"  Done: {remote_path}")
            success_count += 1
        except Exception as e:
            print(f"  [LOI] Upload that bai: {e}")

    # Upload main.py
    if success_count > 0 and os.path.exists(MAIN_SCRIPT):
        print(f" Dang upload [main.py]...")
        try:
            sftp.put(MAIN_SCRIPT, f"{REMOTE_DIR}/main.py")
            print(f"  Done: {REMOTE_DIR}/main.py")
        except Exception as e:
            print(f"  [LOI] Upload main.py that bai: {e}")

    sftp.close()
    client.close()

    if success_count > 0:
        
        print(f"\nDang nhap MaixVision va chay:")
        print(f"  exec(open('/root/models/main.py').read())")
    else:
        print("\n[LOI] Khong upload duoc model nao!")
        print("\nHuong dan tai model YOLOv8 face:")
        print(f"  1. Tren trinh duyet: https://wiki.sipeed.com/maixpy/doc/zh/vision/face_detection.html")
        print(f"  2. Tai yolov8n_face.mud ve")
        print(f"  3. Dat vao: {LOCAL_DIR}/")
        print(f"  4. Chay lai script nay")


if __name__ == "__main__":
    main()
