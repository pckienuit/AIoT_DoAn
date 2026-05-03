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
    client.connect(HOST, port=PORT, username=USER, password=PASS)
    
    commands = [
        ("Check AI Env", "if [ -d 'ai_env' ]; then ai_env/bin/python -c \"import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')\" 2>&1; else echo 'No ai_env folder found.'; fi")
    ]
    
    for name, cmd in commands:
        print(f"\n[{name}]")
        stdin, stdout, stderr = client.exec_command(cmd)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if out:
            print(out)
        if err:
            print(f"Error: {err}")
finally:
    client.close()
