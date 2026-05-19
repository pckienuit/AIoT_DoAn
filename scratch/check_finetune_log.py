import os
import paramiko
from dotenv import load_dotenv

load_dotenv()
HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

stdin, stdout, stderr = client.exec_command("tail -n 30 /root/finetune_recognize.log")
log_out = stdout.read().decode('utf-8')
with open('scratch/ft_log.txt', 'w', encoding='utf-8') as f:
    f.write(log_out)
client.close()
