#!/usr/bin/env python3
import paramiko
import sys

HOST = "10.154.35.1"
USER = "root"
PASS = "root"

SCRIPT = """
from maix import camera
print("--- Camera constructor doc ---")
print(camera.Camera.__init__.__doc__)
print("--- Camera class doc ---")
print(camera.Camera.__doc__)
"""

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, username=USER, password=PASS, timeout=10)
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    sftp = client.open_sftp()
    with sftp.open('/root/print_cam_doc.py', 'w') as f:
        f.write(SCRIPT)
    sftp.close()

    _, stdout, _ = client.exec_command("python3 /root/print_cam_doc.py")
    sys.stdout.buffer.write(stdout.read())
    client.exec_command("rm -f /root/print_cam_doc.py")
    client.close()

if __name__ == "__main__":
    main()
