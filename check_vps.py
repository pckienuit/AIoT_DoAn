"""
check_vps.py — Kết nối VPS, liệt kê file .pth và hỏi user muốn download file nào.
"""
import paramiko
import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

def main():
    print(f"Connecting to {HOST}:{PORT} as {USER}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS)
    print("Connected!")

    # List all .pth files in /root
    stdin, stdout, stderr = client.exec_command("find /root -maxdepth 1 -name '*.pth' -type f 2>/dev/null | sort")
    files = [f.strip() for f in stdout.read().decode().split('\n') if f.strip()]

    print(f"\nFound {len(files)} .pth file(s) on VPS:")
    for i, f in enumerate(files, 1):
        print(f"  [{i}] {f}")

    client.close()

if __name__ == '__main__':
    main()
