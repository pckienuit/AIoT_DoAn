#!/usr/bin/env python3
"""
run_live_mjpeg.py — Runs main.py on MaixCAM, streams console prints, and informs how to view stream.
"""
import paramiko
import sys
import time
import socket

HOST = "10.154.35.1"
USER = "root"
PASS = "root"

def main():
    print("=" * 60)
    print("AIoT Face Lookup: MaixCAM Live Video Stream Console")
    print("=" * 60)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, username=USER, password=PASS, timeout=10)
        print("[1] Connected to MaixCAM via SSH.")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    print("[2] Releasing camera (killing maixvision_server)...")
    client.exec_command("killall -9 maixvision_server; killall -9 python3")
    time.sleep(1)

    print("[3] Launching main.py with MJPEG Stream Server on port 8080...")
    transport = client.get_transport()
    channel = transport.open_session()
    channel.get_pty()
    # Run with -u to make prints unbuffered so we see them instantly
    channel.exec_command("cd /root && python3 -u main.py")

    channel.setblocking(False)

    print("\n" + "#" * 58)
    print(f"   -> Live Camera Feed: http://{HOST}:8080/ ")
    print("   Open this URL in your web browser to view the video stream!")
    print("#" * 58 + "\n")
    print("Streaming console logs (Press Ctrl+C to stop)...")
    print("-" * 60)

    try:
        while True:
            if channel.recv_ready():
                data = channel.recv(4096).decode('utf-8', errors='ignore')
                sys.stdout.write(data)
                sys.stdout.flush()

            if channel.recv_stderr_ready():
                data = channel.recv_stderr(4096).decode('utf-8', errors='ignore')
                sys.stderr.write(data)
                sys.stderr.flush()

            if channel.exit_status_ready():
                print(f"\n[!] Device process exited with status: {channel.recv_exit_status()}")
                break

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping streaming console...")
    finally:
        # Clean up session
        channel.close()
        client.close()
        print("=" * 60)

if __name__ == "__main__":
    main()
