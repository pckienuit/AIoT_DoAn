"""
monitor_phase3.py - Monitor Phase 3 training progress on VPS in real-time.

Run this anytime to check current status:
  python scratch/monitor_phase3.py
"""
import os, sys, time, paramiko
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
load_dotenv()

HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

REMOTE_LOG = "/root/finetune_phase3.log"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

    def run(cmd, timeout=15):
        stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        return stdout.read().decode('utf-8', errors='replace').strip()

    print("=" * 65)
    print("  PHASE 3 TRAINING MONITOR")
    print("=" * 65)

    # Process status
    proc = run("pgrep -a -f 'finetune_phase3' || echo 'NOT RUNNING'")
    print(f"\n[Process] {proc}")

    # GPU
    gpu = run("nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo 'No GPU info'")
    print(f"[GPU]     {gpu}")

    # Log stats
    lines = run(f"wc -l {REMOTE_LOG} 2>/dev/null || echo '0'")
    print(f"[Log]     {lines} lines total in {REMOTE_LOG}")

    # All LFW eval lines
    print("\n--- LFW Eval Results So Far ---")
    lfw_lines = run(f"grep -i 'LFW\\|best model\\|early stop\\|phase 3\\|Done' {REMOTE_LOG} 2>/dev/null | tail -20")
    if lfw_lines:
        print(lfw_lines)
    else:
        print("  (no eval results yet)")

    # All epoch summaries
    print("\n--- Epoch Loss Summaries ---")
    epoch_lines = run(f"grep -i 'Epoch \\[' {REMOTE_LOG} 2>/dev/null | tail -15")
    if epoch_lines:
        print(epoch_lines)
    else:
        print("  (no epoch summaries yet)")

    # Last 20 lines for live context
    print("\n--- Last 20 Log Lines ---")
    tail = run(f"tail -20 {REMOTE_LOG} 2>/dev/null || echo 'Log not found'", timeout=10)
    print(tail)

    client.close()
    print("\n" + "=" * 65)

if __name__ == "__main__":
    main()
