import paramiko
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

def check_progress():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print(f"Connecting to VPS ({HOST})...")
        client.connect(HOST, port=PORT, username=USER, password=PASS)
        print("Connected.\n")
        
        # 1. Check if train.py is running
        print("--- Process Status ---")
        stdin, stdout, stderr = client.exec_command("ps -ef | grep '[p]ython3 train.py'")
        processes = stdout.read().decode().strip()
        if processes:
            print(f"Training is RUNNING:\n{processes}")
        else:
            print("Training is NOT running (or not started with 'python3 train.py').")
        
        # 2. Check for Screen/Tmux sessions
        print("\n--- Multiplexer Sessions ---")
        stdin, stdout, stderr = client.exec_command("screen -ls")
        screen_out = stdout.read().decode().strip()
        if "No Sockets found" not in screen_out and screen_out:
            print(f"Screen sessions found:\n{screen_out}")
        else:
            print("No Screen sessions.")
            
        stdin, stdout, stderr = client.exec_command("tmux ls")
        tmux_out = stdout.read().decode().strip()
        if "no server running" not in stderr.read().decode() and tmux_out:
            print(f"Tmux sessions found:\n{tmux_out}")
        else:
            print("No Tmux sessions.")

        # 3. Check for log files and show last lines
        print("\n--- Training Logs ---")
        log_files = ["train.log", "output.txt", "nohup.out", "log.txt"]
        found_logs = []
        
        # Also check for any .log or .out files in root
        stdin, stdout, stderr = client.exec_command("ls -lh *.log *.out output.txt 2>/dev/null")
        ls_out = stdout.read().decode().strip()
        if ls_out:
            print(f"Potential log files found:\n{ls_out}")
            for line in ls_out.split('\n'):
                parts = line.split()
                if len(parts) > 0:
                    found_logs.append(parts[-1])
        
        if not found_logs:
            print("No log files found automatically.")
        else:
            # Avoid duplicates and non-log files if any
            unique_logs = sorted(list(set(found_logs)))
            for log in unique_logs:
                print(f"\nLast 10 lines of {log}:")
                stdin, stdout, stderr = client.exec_command(f"tail -n 10 {log}")
                print(stdout.read().decode())

        # 4. Check model checkpoint
        print("--- Model Checkpoints ---")
        checkpoint_file = "face_detect_model_vps_finetune.pth"
        stdin, stdout, stderr = client.exec_command(f"ls -lh {checkpoint_file}")
        cp_out = stdout.read().decode().strip()
        if cp_out:
            print(f"Checkpoint found: {cp_out}")
            # Get last modified time
            stdin, stdout, stderr = client.exec_command(f"stat -c %y {checkpoint_file}")
            mtime = stdout.read().decode().strip()
            print(f"Last updated: {mtime}")
        else:
            print(f"Checkpoint '{checkpoint_file}' not found yet.")

        # 5. Resource usage
        print("\n--- Resource Usage ---")
        stdin, stdout, stderr = client.exec_command("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo 'No GPU'")
        gpu_out = stdout.read().decode().strip()
        print(f"GPU Utilization: {gpu_out}")
        
        stdin, stdout, stderr = client.exec_command("free -h | grep Mem")
        mem_out = stdout.read().decode().strip()
        print(f"RAM Usage: {mem_out}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()
        print("\nDisconnected.")

if __name__ == "__main__":
    check_progress()
