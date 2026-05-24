#!/usr/bin/env python3
"""
deploy_to_device.py — Deploy MaixCAM_App files to device via Paramiko (SSH/SFTP).

Usage:
    python deploy_to_device.py [--host 10.154.35.1] [--user root] [--password root]

Requires: paramiko  (already in .venv)
"""
import argparse
import os
import sys

import paramiko

DEVICE_HOST = "10.154.35.1"
DEVICE_USER = "root"
DEVICE_PASS = "root"      # MaixCAM default — change if different
DEVICE_PORT = 22

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# (local relative path, remote absolute path)
FILES = [
    ("MaixCAM_App/config.py",     "/root/config.py"),
    ("MaixCAM_App/sync_cache.py", "/root/sync_cache.py"),
    ("MaixCAM_App/display.py",    "/root/display.py"),
    ("MaixCAM_App/main.py",       "/root/main.py"),
    ("MaixCAM_App/config.json",   "/root/config.json"),
]


def make_client(host: str, user: str, password: str, port: int) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password, timeout=15)
    return client


def ssh_run(client: paramiko.SSHClient, cmd: str) -> tuple[int, str]:
    _, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode() + stderr.read().decode()
    rc  = stdout.channel.recv_exit_status()
    return rc, out.strip()


def sftp_upload(client: paramiko.SSHClient, local: str, remote: str) -> bool:
    local_abs = os.path.join(SCRIPT_DIR, local)
    try:
        sftp = client.open_sftp()
        sftp.put(local_abs, remote)
        sftp.close()
        return True
    except Exception as e:
        print("  ERROR:", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Deploy MaixCAM_App to device")
    parser.add_argument("--host",     default=DEVICE_HOST)
    parser.add_argument("--user",     default=DEVICE_USER)
    parser.add_argument("--password", default=DEVICE_PASS)
    parser.add_argument("--port",     type=int, default=DEVICE_PORT)
    args = parser.parse_args()

    print("=" * 55)
    print("Deploying to {}@{}:{}".format(args.user, args.host, args.port))
    print("=" * 55)

    # ── 1. Connect ──────────────────────────────────────────────────
    print("\n[1] Connecting via SSH...")
    try:
        client = make_client(args.host, args.user, args.password, args.port)
        print("    Connected!")
    except Exception as e:
        print("    FAILED:", e)
        sys.exit(1)

    # ── 2. Device info ──────────────────────────────────────────────
    print("\n[2] Device info:")
    rc, out = ssh_run(client, "uname -a && python3 --version 2>&1")
    for line in out.splitlines():
        print("   ", line)

    # ── 3. Create dirs ──────────────────────────────────────────────
    print("\n[3] Creating /root/cache ...")
    ssh_run(client, "mkdir -p /root/cache")
    print("    Done")

    # ── 4. Upload files ─────────────────────────────────────────────
    print("\n[4] Uploading files...")
    failed = []
    for local, remote in FILES:
        ok = sftp_upload(client, local, remote)
        status = "OK  " if ok else "FAIL"
        print("  [{}] {} -> {}".format(status, local, remote))
        if not ok:
            failed.append(local)

    # ── 5. Syntax check ─────────────────────────────────────────────
    print("\n[5] Python syntax check on device:")
    for local, remote in FILES:
        if not remote.endswith(".py"):
            continue
        rc, out = ssh_run(client, "python3 -m py_compile {} 2>&1 && echo SYNTAX_OK".format(remote))
        ok = "SYNTAX_OK" in out
        print("  [{}] {}".format("OK  " if ok else "ERR ", remote))
        if not ok and out:
            print("       ->", out[:200])

    # ── 6. Verify config.json ────────────────────────────────────────
    print("\n[6] Config on device:")
    rc, out = ssh_run(client, "cat /root/config.json")
    print(out)

    # ── 7. List /root ────────────────────────────────────────────────
    print("\n[7] /root/ file listing:")
    rc, out = ssh_run(client, "ls -lh /root/*.py /root/*.json /root/cache/ 2>/dev/null")
    print(out)

    client.close()

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    if failed:
        print("WARNING: Failed uploads:", failed)
    else:
        print("All files deployed successfully!")

    print("\nNext steps:")
    print("  Run app   : ssh {}@{} 'cd /root && python3 main.py'".format(args.user, args.host))
    print("  Sync now  : ssh {}@{} 'touch /root/sync_now.flag'".format(args.user, args.host))
    print("  Set flight: ssh {}@{} 'echo 1 > /root/active_flight.txt'".format(args.user, args.host))


if __name__ == "__main__":
    main()
