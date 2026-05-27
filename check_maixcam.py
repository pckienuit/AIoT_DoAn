#!/usr/bin/env python3
import paramiko
import time

HOST = "10.154.36.1"
USER = "root"
PASSWORD = "root"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

print("Waiting 35s for camera + VENC init...")
time.sleep(35)

print("=== LOG (last 15 lines) ===")
stdin, stdout, stderr = ssh.exec_command("tail -15 /root/main.log")
log = stdout.read().decode()
print(log)

print("=== VENC ERRORS? ===")
stdin, stdout, stderr = ssh.exec_command("grep -c 'CVI_VENC_GetStream failed' /root/main.log 2>/dev/null")
count = stdout.read().decode().strip()
print(f"VENC errors: {count}")

print("\n=== PROCESS ===")
stdin, stdout, stderr = ssh.exec_command("ps aux | grep main.py | grep -v grep")
print(stdout.read().decode().strip())

print("\n=== PORT 8080 ===")
stdin, stdout, stderr = ssh.exec_command("ss -tlnp | grep 8080")
port = stdout.read().decode().strip()
print(port if port else "(NOT listening)")

ssh.close()
