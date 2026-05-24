#!/usr/bin/env python3
import paramiko
import sys

HOST = "10.154.35.1"
USER = "root"
PASS = "root"

SCRIPT = """
from maix import image
img = image.Image(100, 100)
print("--- to_jpeg doc ---")
print(img.to_jpeg.__doc__)
print("--- to_bytes doc ---")
print(img.to_bytes.__doc__)
print("--- compress doc ---")
print(img.compress.__doc__)
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
    with sftp.open('/root/print_doc.py', 'w') as f:
        f.write(SCRIPT)
    sftp.close()

    _, stdout, _ = client.exec_command("python3 /root/print_doc.py")
    sys.stdout.buffer.write(stdout.read())
    client.exec_command("rm -f /root/print_doc.py")
    client.close()

if __name__ == "__main__":
    main()
