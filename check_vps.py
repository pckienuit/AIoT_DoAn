import paramiko

import os
from dotenv import load_dotenv

load_dotenv()
HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    print("Connecting to VPS...")
    client.connect(HOST, port=PORT, username=USER, password=PASS)
    print("Connected.")
    
    commands = [
        ("CPU Model", "lscpu | grep 'Model name'"),
        ("CPU Cores", "nproc"),
        ("RAM", "free -h"),
        ("GPU", "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'No NVIDIA GPU'"),
        ("Disk", "df -h /"),
        ("OS", "cat /etc/os-release | head -n 5"),
        ("Python version", "python3 --version"),
        ("CUDA", "nvcc --version 2>/dev/null || echo 'nvcc not found'")
    ]
    
    for name, cmd in commands:
        print(f"\n--- {name} ---")
        stdin, stdout, stderr = client.exec_command(cmd)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if out:
            print(out)
        if err:
            print(f"Error: {err}")
finally:
    client.close()
