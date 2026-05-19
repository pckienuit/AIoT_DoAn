import os, time, paramiko
from dotenv import load_dotenv

load_dotenv()
HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

LOCAL_SCRIPT  = "scratch/finetune_recognize.py"
REMOTE_SCRIPT = "/root/finetune_recognize.py"

def run(client, cmd, title):
    print(f"\n[{title}] -> {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode('utf-8').strip()
    err = stderr.read().decode('utf-8').strip()
    if out: print(out)
    if err: print("ERR:", err)
    return out

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {HOST}...")
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    sftp = client.open_sftp()

    # 1. Upload script
    print("Uploading finetune_recognize.py...")
    sftp.put(LOCAL_SCRIPT, REMOTE_SCRIPT)
    sftp.close()

    # 2. Kill train cu (neu co)
    run(client, "pkill -f train_recognize.py || true", "Kill train cu")
    run(client, "pkill -f finetune_recognize.py || true", "Kill finetune cu")
    time.sleep(2)

    # 3. Khoi dong train Phase 2
    cmd_train = (
        "nohup /root/ai_env/bin/python3 /root/finetune_recognize.py "
        "> /root/finetune_recognize.log 2>&1 &"
    )
    run(client, cmd_train, "Khoi dong train (Phase 2 - Warm Restart)")
    time.sleep(5)

    # 4. Kiem tra process
    out = run(client, "pgrep -a -f finetune_recognize.py", "Process check")
    if out: print("OK - Dang chay Phase 2!")
    else:   print("CANH BAO - Khong thay process!")

    # 5. Log dau tien
    run(client, "tail -20 /root/finetune_recognize.log", "Log ban dau")
    client.close()

if __name__ == "__main__":
    main()
