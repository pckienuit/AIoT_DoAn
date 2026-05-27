#!/usr/bin/env python3
"""Force restart MaixCAM app cleanly."""
import paramiko
import time

HOST = "10.154.36.1"
USER = "root"
PASSWORD = "root"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

def run(cmd, wait=2):
    print(f"$ {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    time.sleep(wait)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(out)
    if err:
        print("[ERR]", err)

# 1. Graceful kill (SIGTERM/15) to allow C++ destructors to release camera/VENC buffers
print("=== Killing all python/main.py processes gracefully (SIGTERM) ===")
run("killall -15 python3 2>/dev/null; killall -15 python 2>/dev/null")
print("Waiting 5s for camera and VENC drivers to release buffers...")
time.sleep(5)

# 2. Check if still alive before force-killing
stdin, stdout, stderr = ssh.exec_command("ps aux | grep main.py | grep -v grep")
alive = stdout.read().decode().strip()
if alive:
    print("\n=== Force killing remaining python processes (SIGKILL) ===")
    run("killall -9 python3 2>/dev/null; killall -9 python 2>/dev/null; sleep 1", wait=2)
else:
    print("\nProcesses exited cleanly. No force kill needed.")

# Check nothing is left
print("\n=== Remaining processes ===")
run("ps aux | grep -E 'python|main' | grep -v grep")

# Start fresh
print("\n=== Starting app ===")
run("nohup python -u /root/main.py > /root/main.log 2>&1 &", wait=3)

# Verify it started
print("\n=== Process check after start ===")
run("ps aux | grep main.py | grep -v grep")

# Show first lines of log
print("\n=== Log output (first 20 lines) ===")
run("head -20 /root/main.log 2>/dev/null")

ssh.close()
print("\nDone. Stream should be at http://10.154.36.1:8080 in ~10 seconds")
