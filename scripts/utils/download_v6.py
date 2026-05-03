"""
download_v6.py — Download v6 weights from VPS
"""
import paramiko
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

def download_file(remote_path, local_path):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS)
    sftp = client.open_sftp()

    # Get file size for progress
    stat = sftp.stat(remote_path)
    total = stat.st_size

    print(f"Downloading {remote_path} ({total/1024/1024:.1f} MB)...")

    def progress_cb(transferred, total):
        downloaded_mb = transferred / 1024 / 1024
        total_mb = total / 1024 / 1024
        bar_len = 40
        filled = int(bar_len * transferred / total)
        print(f"\r  [{'='*filled}{' '*(bar_len-filled)}] {downloaded_mb:.1f}/{total_mb:.1f} MB", end='', flush=True)

    sftp.get(remote_path, local_path, callback=progress_cb)
    print(f"\n  -> Saved as {local_path}")

    sftp.close()
    client.close()

if __name__ == '__main__':
    # Download v6 weights
    download_file(
        "/root/face_detect_model_vps_finetune_v6_weights.pth",
        "d:\\AIoT_DoAn\\face_detect_model_vps_finetune_v6.pth"
    )
