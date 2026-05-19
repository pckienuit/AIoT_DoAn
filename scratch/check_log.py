import os, paramiko, time, sys
from dotenv import load_dotenv

load_dotenv()
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(os.getenv('VPS_HOST'), port=int(os.getenv('VPS_PORT', 22)),
          username=os.getenv('VPS_USER'), password=os.getenv('VPS_PASS'))
time.sleep(12)
_, o, _ = c.exec_command('tail -25 /root/recognize_train.log')
log = o.read().decode()
c.close()

# Ghi ra file de doc
with open('scratch/last_log.txt', 'w', encoding='utf-8') as f:
    f.write(log)
print("Done. Xem scratch/last_log.txt")
