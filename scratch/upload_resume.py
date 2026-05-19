"""
upload_resume.py - Upload model checkpoint len VPS va restart train tu checkpoint.
"""
import os, time, paramiko
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

LOCAL_MODEL  = "models/checkpoints/face_recognize_arcface.pth"
REMOTE_MODEL = "/root/models/checkpoints/face_recognize_arcface.pth"

def run(client, cmd, label=""):
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if label:
        print(f"\n[{label}]")
    if out: print(out)
    if err and "no process" not in err.lower(): print(f"  [!] {err}")
    return out

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    sftp = client.open_sftp()

    # 1. Upload checkpoint len VPS
    size = os.path.getsize(LOCAL_MODEL)
    print(f"Upload checkpoint ({size/1024**2:.1f} MB) len VPS...")
    with tqdm(total=size, unit='B', unit_scale=True, desc="  checkpoint") as pbar:
        def _cb(t, _): pbar.n = t; pbar.refresh()
        sftp.put(LOCAL_MODEL, REMOTE_MODEL, callback=_cb)
    print(f"  OK -> {REMOTE_MODEL}")
    sftp.close()

    # 2. Kill train cu
    run(client, "pkill -f train_recognize.py || true", "Kill train cu")
    time.sleep(2)

    # 3. Khoi dong train moi voi RESUME=True (da co checkpoint)
    cmd_train = (
        "nohup /root/ai_env/bin/python3 /root/train_recognize.py "
        "> /root/recognize_train.log 2>&1 &"
    )
    run(client, cmd_train, "Khoi dong train (resume)")
    time.sleep(5)

    # 4. Kiem tra process
    out = run(client, "pgrep -a -f train_recognize.py", "Process check")
    if out: print("OK - Dang chay!")
    else:   print("CANH BAO - Khong thay process!")

    # 5. Log dau tien
    run(client, "tail -20 /root/recognize_train.log", "Log ban dau")
    client.close()

if __name__ == "__main__":
    main()
