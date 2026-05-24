#!/usr/bin/env python3
import paramiko
import sys
import socket

HOST = "10.154.35.1"
USER = "root"
PASS = "root"

SCRIPT = """
import time
import sys

print("--- [1] Initializing Camera ---")
try:
    from maix import camera
    cam = camera.Camera(320, 224)
    print("  Camera init success!")
except Exception as e:
    print("  Camera init failed:", e)

print("--- [2] Initializing Display ---")
try:
    from maix import display
    disp = display.Display()
    print("  Display init success!")
except Exception as e:
    print("  Display init failed:", e)

print("--- [3] Done ---")
"""

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, username=USER, password=PASS, timeout=10)
        print("Connected!")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    sftp = client.open_sftp()
    with sftp.open('/root/test_hw_init.py', 'w') as f:
        f.write(SCRIPT)
    sftp.close()

    print("\nRunning hardware initialization test on device...")
    stdin, stdout, stderr = client.exec_command("python3 -u /root/test_hw_init.py", timeout=10)
    
    try:
        print(stdout.read().decode())
    except socket.timeout:
        print("TIMEOUT: Command hung! Received stdout so far:")
        if stdout.channel.recv_ready():
            print(stdout.channel.recv(4096).decode('utf-8', errors='ignore'))
        else:
            print("  (No stdout received)")
    except Exception as e:
        print("Error:", e)

    client.exec_command("rm -f /root/test_hw_init.py")
    client.close()

if __name__ == "__main__":
    main()
