#!/usr/bin/env python3
import paramiko
import sys

HOST = "10.154.35.1"
USER = "root"
PASS = "root"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, username=USER, password=PASS, timeout=10)
        print("Connected!")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    print("\n--- Listing all processes ---")
    stdin, stdout, stderr = client.exec_command("ps w")
    proc_list = stdout.read().decode('utf-8', errors='ignore')
    
    python_pids = []
    for line in proc_list.splitlines():
        if "python" in line or "main.py" in line or "test_hw" in line:
            print("Found process:", line.strip())
            parts = line.strip().split()
            if parts:
                pid = parts[0]
                python_pids.append((pid, line.strip()))
                
    if python_pids:
        print(f"\nFound {len(python_pids)} python-related processes. Attempting to kill them...")
        for pid, desc in python_pids:
            print(f"Killing PID {pid}...")
            client.exec_command(f"kill -9 {pid}")
        print("Kill commands sent. Checking again in 2 seconds...")
        import time
        time.sleep(2)
        
        _, stdout_check, _ = client.exec_command("ps w")
        proc_list_check = stdout_check.read().decode('utf-8', errors='ignore')
        still_running = False
        for line in proc_list_check.splitlines():
            if "python" in line or "main.py" in line:
                print("Still running:", line.strip())
                still_running = True
        if not still_running:
            print("All python processes successfully terminated!")
    else:
        print("No python processes running.")

    client.close()

if __name__ == "__main__":
    main()
