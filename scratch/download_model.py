import os
import paramiko
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

def create_ssh():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    return client

def download_file(sftp, remote_path, local_path):
    try:
        size = sftp.stat(remote_path).st_size
    except FileNotFoundError:
        print(f"File not found on VPS: {remote_path}")
        return False
        
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    desc = os.path.basename(local_path)
    
    with tqdm(total=size, unit='B', unit_scale=True, desc=f"Downloading {desc}") as pbar:
        def _cb(transferred, total):
            pbar.n = transferred
            pbar.refresh()
        sftp.get(remote_path, local_path, callback=_cb)
    return True

if __name__ == "__main__":
    print(f"Connecting to VPS {USER}@{HOST}:{PORT} ...")
    client = create_ssh()
    sftp = client.open_sftp()
    
    remote_path = "/root/models/checkpoints/face_recognize_arcface.pth"
    local_path = "models/checkpoints/face_recognize_arcface.pth"
    
    success = download_file(sftp, remote_path, local_path)
    
    sftp.close()
    client.close()
    if success:
        print("✅ Download complete!")
