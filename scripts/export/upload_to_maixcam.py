"""
Upload the complete MaixCAM v9 + ArcFace P3 face pipeline via SSH/SFTP.

Expected local model files:
- models/exports/yolov8n_face.mud
- models/exports/yolov8n_face_320.cvimodel
- models/exports/face_detect_v9.mud
- models/exports/face_detect_v9.cvimodel
- models/exports/face_recognize_arcface_p3.mud
- models/exports/face_recognize_arcface_p3.cvimodel
"""

import os

import paramiko

MAIXCAM_HOST = os.getenv("MAIXCAM_HOST", "10.154.36.1")
MAIXCAM_PORT = int(os.getenv("MAIXCAM_PORT", "22"))
MAIXCAM_USER = os.getenv("MAIXCAM_USER", "root")
MAIXCAM_PASS = os.getenv("MAIXCAM_PASS", "root")

LOCAL_DIR = os.path.join("models", "exports")
REMOTE_DIR = "/root/models"

MODELS = [
    ("yolov8n_face.mud", "YOLOv8 face detector metadata"),
    ("yolov8n_face_320.cvimodel", "YOLOv8 face detector cvimodel"),
    ("yolo11s_face.mud", "YOLO11 face detector metadata for MaixCAM2"),
    ("yolo11s_face.cvimodel", "YOLO11 face detector cvimodel for MaixCAM2"),
    ("face_detect_v9.mud", "v9 landmark/refinement metadata"),
    ("face_detect_v9.cvimodel", "v9 landmark/refinement cvimodel"),
    ("face_recognize_arcface_p3.mud", "ArcFace P3 recognition metadata"),
    ("face_recognize_arcface_p3.cvimodel", "ArcFace P3 recognition cvimodel"),
]

MAIN_SCRIPT = os.path.join("MaixCAM_App", "main.py")


def upload_file(sftp, local_path, remote_path, desc):
    if not os.path.exists(local_path):
        print(f"[SKIP] Missing {desc}: {local_path}")
        return False
    print(f"Uploading {desc}...")
    sftp.put(local_path, remote_path)
    print(f"  -> {remote_path}")
    return True


def main():
    print(f"Connecting to MaixCAM {MAIXCAM_HOST}:{MAIXCAM_PORT}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(
            MAIXCAM_HOST,
            port=MAIXCAM_PORT,
            username=MAIXCAM_USER,
            password=MAIXCAM_PASS,
            timeout=10,
        )
    except Exception as e:
        print(f"[ERROR] Cannot connect: {e}")
        return

    sftp = client.open_sftp()
    client.exec_command(f"mkdir -p {REMOTE_DIR}")

    uploaded = 0
    for model_file, desc in MODELS:
        local_path = os.path.join(LOCAL_DIR, model_file)
        remote_path = f"{REMOTE_DIR}/{model_file}"
        if upload_file(sftp, local_path, remote_path, desc):
            uploaded += 1

    if os.path.exists(MAIN_SCRIPT):
        upload_file(sftp, MAIN_SCRIPT, "/root/main.py", "MaixCAM app")
    else:
        print(f"[ERROR] Missing app: {MAIN_SCRIPT}")

    sftp.close()
    client.close()

    print(f"\nUploaded {uploaded} model files.")
    print("Run on MaixCAM / MaixVision:")
    print("  exec(open('/root/main.py').read())")
    print("\nRegister identity while app is running:")
    print("  echo Kien > /root/register_name.txt")
    print("\nClear identity database:")
    print("  touch /root/clear_face_db.flag")


if __name__ == "__main__":
    main()
