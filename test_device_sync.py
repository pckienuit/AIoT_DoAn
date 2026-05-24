#!/usr/bin/env python3
"""
test_device_sync.py — Test device can reach server and sync flight cache.

Run: python test_device_sync.py
"""
import paramiko

HOST = "10.154.35.1"
USER = "root"
PASS = "root"

TEST_SCRIPT = """
import sys
sys.path.insert(0, '/root')
from config import load_config
from sync_cache import CacheManager
import json

cfg = load_config()
print("[config] server:", cfg['server_url'])

cm = CacheManager(cfg)

# Test sync flight 1
n = cm.sync(1)
print("[sync] result:", n)

# Check cache info
info = cm.get_cache_info(1)
print("[cache] info:", json.dumps(info, default=str))

# List cache directory
import os
files = os.listdir('/root/cache')
print("[cache] files:", files)
"""


def ssh_run(client, cmd):
    _, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode()
    err = stderr.read().decode()
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def main():
    print("=" * 55)
    print("Testing device -> server sync")
    print("=" * 55)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=10)
    print("[1] SSH connected to", HOST)

    # Upload and run test script
    sftp = client.open_sftp()
    with sftp.open('/root/_test_sync.py', 'w') as f:
        f.write(TEST_SCRIPT)
    sftp.close()

    print("[2] Running sync test on device...")
    rc, out, err = ssh_run(client, 'cd /root && python3 /root/_test_sync.py')

    print("\n--- Output ---")
    print(out)
    if err:
        print("--- Stderr ---")
        print(err)

    # Cleanup
    ssh_run(client, 'rm -f /root/_test_sync.py')
    client.close()

    print("\n" + "=" * 55)
    if "cached" in out and "synced_at" in out:
        print("PASS: Device -> Server sync pipeline working!")
        print("      (DB is empty — add bookings via web_stage3 to populate)")
    else:
        print("FAIL: Check server IP / connectivity")
    print("=" * 55)


if __name__ == "__main__":
    main()
