#!/usr/bin/env python3
"""
run_live_test.py — Non-blocking wrapper to launch main.py on MaixCAM and check output logs.
"""
import paramiko
import sys
import time

HOST = "10.154.35.1"
USER = "root"
PASS = "root"

def main():
    print("=" * 60)
    print("MaixCAM Live Execution Wrapper")
    print("=" * 60)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, username=USER, password=PASS, timeout=10)
        print("[1] Connected to MaixCAM via SSH.")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    # Clean up and prepare log file
    print("[2] Killing conflicting processes (python3, maixvision_server)...")
    client.exec_command("killall -9 python3; killall -9 maixvision_server; rm -f /root/app.log")
    time.sleep(1)

    # Launch app in background
    print("[3] Launching /root/main.py in background...")
    # Using proper nohup redirection so it doesn't block the SSH channel
    stdin, stdout, stderr = client.exec_command(
        "nohup python3 /root/main.py > /root/app.log 2>&1 &"
    )
    # Wait for the background launch command to return
    stdout.channel.recv_exit_status()

    # Sleep to allow app startup
    print("[4] Sleeping for 8 seconds to allow camera & models initialization...")
    time.sleep(8)

    # Read log file
    print("[5] Fetching /root/app.log content:")
    print("-" * 50)
    _, log_out, _ = client.exec_command("cat /root/app.log")
    print(log_out.read().decode('utf-8', errors='ignore').strip())
    print("-" * 50)

    # Check running status
    _, ps_out, _ = client.exec_command("ps w | grep python")
    processes = ps_out.read().decode().strip()
    print("[6] Current active Python processes:")
    print(processes)

    if "/root/main.py" in processes:
        print("\nSUCCESS: main.py is running on MaixCAM! View the device LCD screen.")
    else:
        print("\nFAILURE: main.py exited. Check the log output above for errors.")

    client.close()
    print("=" * 60)

if __name__ == "__main__":
    main()
