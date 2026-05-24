#!/usr/bin/env python3
"""
test_image_methods.py — Inspect image conversion methods on MaixCAM.
"""
import paramiko
import sys

HOST = "10.154.35.1"
USER = "root"
PASS = "root"

SCRIPT = """
from maix import image
img = image.Image(100, 100)
print("Image methods containing jpeg, encode, compress or format:")
from maix import image
img = image.Image(100, 100)
from maix import image
img = image.Image(100, 100)
try:
    res = img.to_format(image.Format.FMT_JPEG)
    print("to_format(JPEG) type:", type(res))
    if res:
        print("  format:", res.format())
        b = res.to_bytes()
        print("  to_bytes() type:", type(b), "len:", len(b))
except Exception as e:
    print("to_format failed:", e)


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
    with sftp.open('/root/inspect_img.py', 'w') as f:
        f.write(SCRIPT)
    sftp.close()

    _, stdout, _ = client.exec_command("python3 /root/inspect_img.py")
    print(stdout.read().decode())
    client.exec_command("rm -f /root/inspect_img.py")
    client.close()

if __name__ == "__main__":
    main()
