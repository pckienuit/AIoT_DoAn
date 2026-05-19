"""
launch_phase3.py - Upload Phase 3 training script and start training on VPS.
"""
import os, sys, time, paramiko
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
load_dotenv()

HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

LOCAL_SCRIPT  = "scratch/finetune_phase3.py"
REMOTE_SCRIPT = "/root/finetune_phase3.py"
REMOTE_LOG    = "/root/finetune_phase3.log"

def run(client, cmd, label="", timeout=15):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    if label:
        print(f"\n[{label}]")
    if out:
        print(out)
    if err and err.strip() and "no process" not in err.lower():
        print(f"  [stderr] {err[:300]}")
    return out

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {HOST}:{PORT}...")
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    sftp = client.open_sftp()
    print("Connected!\n")

    # 1. Kill any existing training process
    run(client,
        "pkill -f 'finetune_phase3\\|finetune_recognize\\|train_recognize' || true",
        "Kill old processes")
    time.sleep(2)

    # 2. Upload Phase 3 script
    print(f"Uploading {LOCAL_SCRIPT} -> {REMOTE_SCRIPT}...")
    sftp.put(LOCAL_SCRIPT, REMOTE_SCRIPT)
    size = sftp.stat(REMOTE_SCRIPT).st_size
    print(f"  OK ({size} bytes)")
    sftp.close()

    # 3. Verify Phase 2 checkpoint exists on VPS
    run(client,
        "ls -lh /root/models/checkpoints/face_recognize_arcface_ft.pth || echo 'ERROR: ft model not found!'",
        "Check Phase2 checkpoint")

    # 4. Launch Phase 3 training in background (nohup)
    cmd_train = (
        f"nohup /root/ai_env/bin/python3 {REMOTE_SCRIPT} "
        f"> {REMOTE_LOG} 2>&1 &"
    )
    run(client, cmd_train, "Launch Phase 3 training")
    time.sleep(5)  # Wait for process to initialize

    # 5. Verify process running
    out = run(client, "pgrep -a -f 'finetune_phase3'", "Check process")
    if out:
        print("  Process running!")
    else:
        print("  WARNING: Process not detected, checking log...")

    # 6. Show initial log output
    time.sleep(5)
    run(client, f"tail -25 {REMOTE_LOG}", "Initial log", timeout=10)

    # 7. GPU check
    run(client,
        "nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader",
        "GPU status")

    client.close()
    print(f"\nPhase 3 training launched!")
    print(f"  Log: {REMOTE_LOG}")
    print(f"  Monitor: python scratch/monitor_phase3.py")

if __name__ == "__main__":
    main()
