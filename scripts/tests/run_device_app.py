#!/usr/bin/env python3
"""
run_device_app.py — Runs main.py on MaixCAM and streams its console output.
"""
import paramiko
import sys
import time

HOST = "10.154.35.1"
USER = "root"
PASS = "root"

def main():
    print(f"Connecting to MaixCAM at {HOST}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, username=USER, password=PASS, timeout=10)
        print("Connected!")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    print("\nStarting main.py on MaixCAM. Press Ctrl+C to terminate.\n")
    transport = client.get_transport()
    channel = transport.open_session()
    channel.get_pty()
    channel.exec_command("cd /root && python3 main.py")

    # Set non-blocking
    channel.setblocking(False)

    try:
        while True:
            # Check stdout
            if channel.recv_ready():
                data = channel.recv(4096).decode('utf-8', errors='ignore')
                sys.stdout.write(data)
                sys.stdout.flush()

            # Check stderr
            if channel.recv_stderr_ready():
                data = channel.recv_stderr(4096).decode('utf-8', errors='ignore')
                sys.stderr.write(data)
                sys.stderr.flush()

            # Check if terminated
            if channel.exit_status_ready():
                print(f"\nDevice process exited with status: {channel.recv_exit_status()}")
                break

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping device process...")
        channel.close()
    finally:
        client.close()

if __name__ == "__main__":
    main()
