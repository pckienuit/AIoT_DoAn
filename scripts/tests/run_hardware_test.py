#!/usr/bin/env python3
"""
run_hardware_test.py — Run hardware init diagnostics in background and print log file.
"""
import paramiko
import sys
import time

HOST = "10.154.35.1"
USER = "root"
PASS = "root"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, username=USER, password=PASS, timeout=10)
        print("[1] Connected to MaixCAM.")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    print("[2] Preparing diag script...")
    # Re-upload the diag script to make sure it's fresh
    from test_hardware_init import DIAG_SCRIPT
    sftp = client.open_sftp()
    with sftp.open('/root/diag_hardware_init.py', 'w') as f:
        f.write(DIAG_SCRIPT)
    sftp.close()

    print("[3] Killing existing Python/Server instances and clearing old log...")
    client.exec_command("killall -9 python3; killall -9 maixvision_server; rm -f /root/diag.log")
    time.sleep(1)

    print("[4] Launching diag script in background...")
    client.exec_command("nohup python3 /root/diag_hardware_init.py > /root/diag.log 2>&1 &")
    time.sleep(8)

    print("[5] Fetching diagnostic log content:")
    print("-" * 50)
    _, stdout, _ = client.exec_command("cat /root/diag.log")
    print(stdout.read().decode('utf-8', errors='ignore').strip())
    print("-" * 50)

    _, ps_out, _ = client.exec_command("ps w | grep python")
    processes = ps_out.read().decode().strip()
    print("[6] Current active Python processes:")
    print(processes)

    client.exec_command("rm -f /root/diag_hardware_init.py")
    client.close()

if __name__ == "__main__":
    main()
