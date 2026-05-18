import os
import paramiko
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

REMOTE_MODEL = "/root/models/checkpoints/face_recognize_arcface.pth"
LOCAL_MODEL = "models/checkpoints/face_recognize_arcface.pth"

def download_file():
    print(f"=== Kéo model Face Recognition từ VPS ===")
    print(f"Server: {USER}@{HOST}:{PORT}")
    
    # Kết nối SSH
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    except Exception as e:
        print(f"❌ Lỗi kết nối VPS: {e}")
        return

    sftp = client.open_sftp()
    
    # Kiểm tra file trên VPS
    try:
        stat = sftp.stat(REMOTE_MODEL)
        size_mb = stat.st_size / (1024 * 1024)
        print(f"✅ Tìm thấy model trên VPS: {size_mb:.2f} MB")
    except FileNotFoundError:
        print(f"❌ KHÔNG TÌM THẤY file: {REMOTE_MODEL} trên VPS!")
        sftp.close()
        client.close()
        return

    # Tạo thư mục local nếu chưa có
    os.makedirs(os.path.dirname(LOCAL_MODEL), exist_ok=True)

    # Tải file
    desc = os.path.basename(LOCAL_MODEL)
    try:
        with tqdm(total=stat.st_size, unit='B', unit_scale=True, desc=f"📥 Tải về") as pbar:
            def _cb(transferred, total):
                pbar.n = transferred
                pbar.refresh()
            sftp.get(REMOTE_MODEL, LOCAL_MODEL, callback=_cb)
        print(f"\n✅ Đã lưu model về: {LOCAL_MODEL}")
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình tải: {e}")
    finally:
        sftp.close()
        client.close()

if __name__ == "__main__":
    download_file()
