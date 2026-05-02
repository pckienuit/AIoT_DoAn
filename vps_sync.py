import paramiko
import sys
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Tải cấu hình từ file .env
load_dotenv()

HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

def create_ssh_client():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS)
    return client

def progress_callback(transferred, total):
    if not hasattr(progress_callback, 'pbar'):
        progress_callback.pbar = tqdm(total=total, unit='B', unit_scale=True, desc="Transferring")
    
    progress_callback.pbar.update(transferred - progress_callback.pbar.n)
    
    if transferred == total:
        progress_callback.pbar.close()
        delattr(progress_callback, 'pbar')

def upload_files():
    print("Connecting to VPS for UPLOAD...")
    client = create_ssh_client()
    sftp = client.open_sftp()
    
    files_to_upload = [
        "train.py",
    ]

    for file in files_to_upload:
        if os.path.exists(file):
            print(f"\nUploading {file}...")
            sftp.put(file, f"/root/{file}", callback=progress_callback)
            print(f"  -> {file} uploaded.")
        else:
            print(f"[Warning] Local file '{file}' not found. Skipping.")
    
    sftp.close()
    
    print("\n--- UPLOAD COMPLETE ---")
    print("Now you can SSH into the VPS and run:")
    print("  source ai_env/bin/activate")
    print("  python3 train.py")
    client.close()

def download_model():
    print("Connecting to VPS for DOWNLOAD...")
    client = create_ssh_client()
    sftp = client.open_sftp()
    
    remote_model = "/root/face_detect_model_vps_finetune_v7.pth"
    local_model  = "face_detect_model_vps_finetune_v7.pth"
    
    try:
        sftp.stat(remote_model)
        print(f"Downloading {remote_model}...")
        sftp.get(remote_model, local_model, callback=progress_callback)
        print(f"  -> Successfully saved as {local_model}")
    except FileNotFoundError:
        print(f"[Error] {remote_model} not found on VPS. Have you finished training?")
        
    sftp.close()
    client.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vps_sync.py [upload|download]")
        sys.exit(1)
        
    action = sys.argv[1].lower()
    if action == "upload":
        upload_files()
    elif action == "download":
        download_model()
    else:
        print("Invalid action. Use 'upload' or 'download'.")
