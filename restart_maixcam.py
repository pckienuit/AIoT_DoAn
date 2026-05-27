#!/usr/bin/env python3
"""Force restart MaixCAM app cleanly.

Sends SIGTERM to trigger the graceful shutdown handler in main.py,
which calls cam.close() / disp.close() to release VENC/ISP hardware
buffers. Waits for the cleanup log line before force-killing.
"""
import paramiko
import time

HOST = "10.154.36.1"
USER = "root"
PASSWORD = "root"

CLEANUP_MARKER = "Cleanup complete. Safe to restart."
GRACEFUL_TIMEOUT = 8  # seconds to wait for graceful shutdown


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
    return out


def wait_for_cleanup(timeout):
    """Poll the log file for the cleanup marker line."""
    print(f"Waiting up to {timeout}s for graceful hardware release...")
    start = time.time()
    while time.time() - start < timeout:
        stdin, stdout, stderr = ssh.exec_command(
            f"grep -c '{CLEANUP_MARKER}' /root/main.log 2>/dev/null"
        )
        time.sleep(0.5)
        count = stdout.read().decode().strip()
        if count.isdigit() and int(count) > 0:
            print(">> Graceful cleanup confirmed in log.")
            return True
    return False


# 1. Truncate log so we only look for NEW cleanup messages
run("truncate -s 0 /root/main.log 2>/dev/null", wait=0)

# 2. Graceful kill (SIGTERM) — triggers signal handler in main.py
#    which calls cam.close(), disp.close(), del models, gc.collect()
print("\n=== Sending SIGTERM to python processes ===")
run("killall -15 python3 2>/dev/null; killall -15 python 2>/dev/null", wait=1)

# 3. Wait for cleanup marker in the log
cleanup_ok = wait_for_cleanup(GRACEFUL_TIMEOUT)

# 4. Check if still alive
stdin, stdout, stderr = ssh.exec_command("ps aux | grep main.py | grep -v grep")
time.sleep(1)
alive = stdout.read().decode().strip()

if alive:
    if not cleanup_ok:
        print("\n[WARN] Cleanup not confirmed — process may not have released hardware.")
    print("\n=== Force killing remaining python processes (SIGKILL) ===")
    run("killall -9 python3 2>/dev/null; killall -9 python 2>/dev/null; sleep 1", wait=2)
    # Give the kernel time to reclaim hardware resources after forced kill
    print("Waiting 3s for kernel to reclaim hardware resources...")
    time.sleep(3)
else:
    print("\nProcesses exited cleanly after SIGTERM.")

# 5. Verify no python processes remain
print("\n=== Remaining processes ===")
run("ps aux | grep -E 'python|main' | grep -v grep")

# 6. Start fresh
print("\n=== Starting app ===")
run("nohup python -u /root/main.py > /root/main.log 2>&1 &", wait=3)

# 7. Verify it started
print("\n=== Process check after start ===")
run("ps aux | grep main.py | grep -v grep")

# 8. Show first lines of log
print("\n=== Log output (first 20 lines) ===")
run("head -20 /root/main.log 2>/dev/null")

ssh.close()
print("\nDone. Stream should be at http://10.154.36.1:8080 in ~10 seconds")
