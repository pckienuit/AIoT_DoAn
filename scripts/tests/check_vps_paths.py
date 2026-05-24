import paramiko, os
from dotenv import load_dotenv
load_dotenv()
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(os.getenv('VPS_HOST'), port=int(os.getenv('VPS_PORT',22)),
               username=os.getenv('VPS_USER'), password=os.getenv('VPS_PASS'))
transport = client.get_transport()
transport.set_keepalive(30)

cmds = [
    # Check if dir still exists somewhere
    'ls -la /root/img_align_celebA/ 2>/dev/null || echo GONE',
    # Check zip integrity
    'unzip -t /root/celebA_dataset.zip 2>&1 | tail -5',
    # Extract in background
    'mkdir -p /root/img_align_celebA && cd /root/img_align_celebA && unzip -q /root/celebA_dataset.zip &',
    'echo "Extraction started in background (PID: $!)"',
]
for cmd in cmds:
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    print(f'> {cmd}')
    if out: print(' ', out[:300])
    if err: print(' ERR:', err[:200])
client.close()
