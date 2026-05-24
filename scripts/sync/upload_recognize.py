"""
upload_recognize.py — Upload train_recognize.py + face_align.py lên VPS
và tạo thư mục CASIAWebFace_dataset trên VPS.

Dataset CASIA đã có sẵn trên VPS (501K ảnh — không upload lại).
Script chỉ upload code, sau đó kích hoạt training.
"""

import os
import sys
import paramiko
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Các file cần upload
FILES_TO_UPLOAD = [
    ("train_recognize.py",          "/root/train_recognize.py"),
    ("scripts/utils/face_align.py", "/root/scripts/utils/face_align.py"),
]

# Remote paths cần tạo nếu chưa có
REMOTE_DIRS = [
    "/root/scripts",
    "/root/scripts/utils",
    "/root/models",
    "/root/models/checkpoints",
    "/root/CASIAWebFace_dataset",
    "/root/CASIAWebFace_dataset/casia-webface",
    "/root/CASIAWebFace_dataset/eval",
]

def create_ssh():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    t = client.get_transport()
    if t:
        t.set_keepalive(30)
    return client


def ensure_remote_dirs(sftp, dirs):
    for d in dirs:
        try:
            sftp.stat(d)
        except FileNotFoundError:
            sftp.mkdir(d)
            print(f"  [mkdir] {d}")


def upload_file(sftp, local_path, remote_path):
    size = os.path.getsize(local_path)
    desc = os.path.basename(local_path)
    with tqdm(total=size, unit='B', unit_scale=True, desc=f"  {desc}") as pbar:
        def _cb(transferred, total):
            pbar.n = transferred
            pbar.refresh()
        sftp.put(local_path, remote_path, callback=_cb)


def check_casia_on_vps(sftp):
    """Kiểm tra xem CASIA dataset đã có trên VPS chưa."""
    paths_to_check = [
        "/root/CASIAWebFace_dataset/casia-webface/train.rec",
        "/root/CASIAWebFace_dataset/casia-webface/train.idx",
        "/root/CASIAWebFace_dataset/casia-webface/property",
    ]
    all_found = True
    for p in paths_to_check:
        try:
            stat = sftp.stat(p)
            print(f"  ✓ {p} ({stat.st_size / 1024**2:.0f} MB)")
        except FileNotFoundError:
            print(f"  ✗ MISSING: {p}")
            all_found = False
    return all_found


def main():
    print(f"=== Upload Face Recognition files to VPS ===")
    print(f"Target: {USER}@{HOST}:{PORT}\n")

    client = create_ssh()
    sftp   = client.open_sftp()

    # 1. Tạo thư mục cần thiết
    print("[1/4] Tạo thư mục remote...")
    ensure_remote_dirs(sftp, REMOTE_DIRS)

    # 2. Kiểm tra CASIA trên VPS
    print("\n[2/4] Kiểm tra CASIA-WebFace trên VPS...")
    casia_ok = check_casia_on_vps(sftp)
    if not casia_ok:
        print("\n  ⚠ CASIA dataset chưa có trên VPS!")
        print("  -> Cần copy thủ công hoặc upload riêng (501K ảnh ~4GB).")
        print("  -> Script train vẫn sẽ upload, nhưng cần dataset trước khi chạy.")

    # 3. Upload các file code
    print("\n[3/4] Upload file code...")
    for local_rel, remote_path in FILES_TO_UPLOAD:
        local_path = os.path.join(ROOT_DIR, local_rel)
        if os.path.exists(local_path):
            upload_file(sftp, local_path, remote_path)
            print(f"  ✓ Uploaded -> {remote_path}")
        else:
            print(f"  ✗ Không tìm thấy local: {local_path}")

    sftp.close()

    # 4. Kiểm tra môi trường Python trên VPS
    print("\n[4/4] Kiểm tra môi trường Python trên VPS...")
    commands_to_check = [
        "python3 -c \"import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())\"",
        "python3 -c \"import torchvision; print('torchvision:', torchvision.__version__)\"",
        "python3 -c \"from PIL import Image; print('Pillow OK')\"",
    ]
    for cmd in commands_to_check:
        stdin, stdout, stderr = client.exec_command(f"source ai_env/bin/activate 2>/dev/null; {cmd}")
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if out:
            print(f"  {out}")
        if err and "not found" not in err.lower():
            print(f"  [!] {err}")

    client.close()

    print("\n=== Upload hoàn tất! ===")
    print("\nĐể bắt đầu train trên VPS, SSH vào và chạy:")
    print(f"  ssh -p {PORT} {USER}@{HOST}")
    print("  source ai_env/bin/activate")
    print("  nohup python3 train_recognize.py > recognize_train.log 2>&1 &")
    print("  tail -f recognize_train.log")


if __name__ == "__main__":
    main()
