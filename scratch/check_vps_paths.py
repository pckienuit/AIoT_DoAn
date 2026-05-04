import paramiko
import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

def check_vps_filesystem():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, port=PORT, username=USER, password=PASS)
        
        commands = [
            "ls -F /root",
            "ls -F /root/img_align_celebA | head -n 5",
            "ls -F /root/img_align_celebA/img_align_celebA | head -n 5",
            "find /root -name \"000001.jpg\""
        ]
        
        for cmd in commands:
            print(f"\n--- Running: {cmd} ---")
            stdin, stdout, stderr = client.exec_command(cmd)
            print(stdout.read().decode().strip())
            
        client.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_vps_filesystem()
