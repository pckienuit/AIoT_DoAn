#!/usr/bin/env python3
"""Upload both updated files to MaixCAM and restart cleanly."""
import paramiko
import time

HOST = "10.154.36.1"
USER = "root"
PASSWORD = "root"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)
sftp = ssh.open_sftp()

# Upload both files
print("[1] Uploading mjpeg_server.py ...")
sftp.put(r"d:\AIoT_DoAn\MaixCAM_App\mjpeg_server.py", "/root/mjpeg_server.py")
print("    Done.")

print("[2] Uploading main.py ...")
sftp.put(r"d:\AIoT_DoAn\MaixCAM_App\main.py", "/root/main.py")
print("    Done.")
sftp.close()

# Kill old processes
print("[3] Killing all python processes ...")
ssh.exec_command("killall -9 python3; killall -9 python; sleep 2")
time.sleep(3)

# Restart with unbuffered output
print("[4] Starting app (python -u) ...")
ssh.exec_command("nohup python -u /root/main.py > /root/main.log 2>&1 &")
time.sleep(3)

# Verify
print("[5] Checking process ...")
stdin, stdout, stderr = ssh.exec_command("ps aux | grep main.py | grep -v grep")
proc = stdout.read().decode().strip()
print("    ", proc if proc else "(not found!)")

print("\nDone! Stream at http://10.154.36.1:8080 in ~30s")
ssh.close()
