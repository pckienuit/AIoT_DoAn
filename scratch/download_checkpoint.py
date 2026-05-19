"""
download_checkpoint.py - Tai model final tu VPS ve may local.
"""
import os, paramiko
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

REMOTE_MODEL = "/root/models/checkpoints/face_recognize_arcface.pth"
LOCAL_MODEL  = "models/checkpoints/face_recognize_arcface.pth"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {HOST}...")
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    sftp = client.open_sftp()

    print(f"Downloading {REMOTE_MODEL} from VPS...")
    os.makedirs(os.path.dirname(LOCAL_MODEL), exist_ok=True)
    
    try:
        remote_size = sftp.stat(REMOTE_MODEL).st_size
        with tqdm(total=remote_size, unit='B', unit_scale=True, desc="  Downloading") as pbar:
            def _cb(t, _): pbar.n = t; pbar.refresh()
            sftp.get(REMOTE_MODEL, LOCAL_MODEL, callback=_cb)
        print(f"  OK -> {LOCAL_MODEL} ({remote_size/1024**2:.1f} MB)")
    except Exception as e:
        print(f"Loi khi download: {e}")
        
    sftp.close()
    client.close()

if __name__ == "__main__":
    main()
