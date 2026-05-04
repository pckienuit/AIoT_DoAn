import paramiko
import sys
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR

def create_ssh_client():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS)
    transport = client.get_transport()
    if transport:
        transport.set_keepalive(30)
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
        ("train_v9.py", "/root/train_v9.py"),
    ]

    for local_name, remote_name in files_to_upload:
        local_path = os.path.join(ROOT_DIR, local_name)
        if os.path.exists(local_path):
            print(f"\nUploading {local_path}...")
            sftp.put(local_path, remote_name, callback=progress_callback)
            print(f"  -> {local_name} uploaded.")
        else:
            print(f"[Warning] Local file '{local_path}' not found. Skipping.")

    sftp.close()
    print("\n--- UPLOAD COMPLETE ---")
    print("Now you can SSH into the VPS and run:")
    print("  source ai_env/bin/activate")
    print("  python3 train_v9.py")
    client.close()


def download_model():
    local_model = os.path.join(ROOT_DIR, "face_detect_model_vps_finetune_v9.pth")
    remote_model = "/root/face_detect_model_vps_finetune_v9.pth"
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks

    def _do_download(client, offset=0):
        sftp = client.open_sftp()
        fstat = sftp.stat(remote_model)
        total_size = fstat.st_size

        mode = 'ab' if offset > 0 else 'wb'
        with open(local_model, mode) as lf:
            if offset > 0:
                lf.seek(offset)
            with sftp.open(remote_model, 'rb') as rf:
                rf.prefetch(total_size)
                rf.seek(offset)
                pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading ({offset//1024**2:.0f}MB+)", initial=offset)
                while True:
                    chunk = rf.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    lf.write(chunk)
                    pbar.update(len(chunk))
                pbar.close()
        sftp.close()
        return total_size

    # Check remote file exists
    client = create_ssh_client()
    sftp_check = client.open_sftp()
    try:
        stat = sftp_check.stat(remote_model)
        print(f"Remote file size: {stat.st_size / 1024**2:.1f} MB")
    except FileNotFoundError:
        print(f"[Error] {remote_model} not found on VPS.")
        sftp_check.close()
        client.close()
        return
    sftp_check.close()

    # Resume from partial download if any
    resume_offset = os.path.getsize(local_model) if os.path.exists(local_model) else 0
    if resume_offset > 0:
        print(f"Resuming from {resume_offset / 1024**2:.1f} MB (attempt 1)...")

    for attempt in range(1, 4):
        try:
            size = _do_download(client, offset=resume_offset)
            downloaded = os.path.getsize(local_model)
            if downloaded >= size:
                print(f"\n  -> Download complete: {local_model} ({downloaded / 1024**2:.1f} MB)")
            else:
                print(f"\n  -> Partial file: {downloaded / 1024**2:.1f} MB / {size / 1024**2:.1f} MB (retry {attempt+1})")
                resume_offset = downloaded
            break
        except (paramiko.SSHException, EOFError, OSError, IOError) as e:
            print(f"\n  Attempt {attempt} failed: {e}")
            resume_offset = os.path.getsize(local_model) if os.path.exists(local_model) else 0
            client.close()
            if attempt < 3:
                print(f"  Reconnecting... ({attempt+1}/3)")
                client = create_ssh_client()
            else:
                print("[Error] All attempts failed.")

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
