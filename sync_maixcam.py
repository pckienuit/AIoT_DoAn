#!/usr/bin/env python3
"""Trigger a manual cache sync on the MaixCAM device."""
import argparse
import sys
import time

import paramiko


DEVICE_HOST = "10.154.36.1"
DEVICE_USER = "root"
DEVICE_PASS = "root"
DEVICE_PORT = 22


def make_client(host: str, user: str, password: str, port: int) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password, timeout=15)
    return client


def ssh_run(client: paramiko.SSHClient, command: str) -> tuple[int, str]:
    _, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    return rc, (out + err).strip()


def quote_shell(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def trigger_flag_sync(client: paramiko.SSHClient, flight_id: int | None, wait_sec: int) -> None:
    if flight_id is not None:
        rc, out = ssh_run(client, "echo {} > /root/active_flight.txt".format(int(flight_id)))
        if rc != 0:
            raise RuntimeError(out or "Failed to set active flight")
        print("[1] Active flight set to {}".format(flight_id))
    else:
        print("[1] Keeping current active flight")

    rc, out = ssh_run(client, "touch /root/sync_now.flag && ls -l /root/sync_now.flag")
    if rc != 0:
        raise RuntimeError(out or "Failed to create sync flag")
    print("[2] Sync flag created: /root/sync_now.flag")

    if wait_sec > 0:
        print("[3] Waiting {}s for MaixCAM app to pick up the flag...".format(wait_sec))
        time.sleep(wait_sec)
    else:
        print("[3] Not waiting for app confirmation")


def run_direct_sync(client: paramiko.SSHClient, flight_id: int | None) -> None:
    if flight_id is None:
        remote_script = r"""
from config import load_config
from sync_cache import CacheManager

cfg = load_config()
cache = CacheManager(cfg)
flight_ids = cfg.get("flight_ids", [1])
print("[direct] Syncing flights:", flight_ids)
print(cache.sync_all(flight_ids))
"""
    else:
        remote_script = r"""
from config import load_config
from sync_cache import CacheManager

cfg = load_config()
cache = CacheManager(cfg)
flight_id = __FLIGHT_ID__
print("[direct] Syncing flight:", flight_id)
print(cache.sync(flight_id))
""".replace("__FLIGHT_ID__", str(int(flight_id)))

    command = "cd /root && python - <<'PY'\n{}\nPY".format(remote_script.strip())
    rc, out = ssh_run(client, command)
    if rc != 0:
        raise RuntimeError(out or "Direct sync failed")
    print(out)


def print_status(client: paramiko.SSHClient, flight_id: int | None) -> None:
    print("\n=== MaixCAM process ===")
    _, out = ssh_run(client, "ps aux | grep main.py | grep -v grep")
    print(out or "(main.py is not running)")

    print("\n=== Cache files ===")
    if flight_id is None:
        command = "ls -lh /root/cache/flight_*.json 2>/dev/null || true"
    else:
        command = "ls -lh /root/cache/flight_{}.json 2>/dev/null || true".format(int(flight_id))
    _, out = ssh_run(client, command)
    print(out or "(no cache file found)")

    print("\n=== Recent sync logs ===")
    _, out = ssh_run(
        client,
        "tail -60 /root/main.log 2>/dev/null | grep -E '\\[sync\\]|Auto-sync|Initial sync|Sync failed' || true",
    )
    print(out or "(no sync log found)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Trigger MaixCAM cache sync manually")
    parser.add_argument("--host", default=DEVICE_HOST)
    parser.add_argument("--user", default=DEVICE_USER)
    parser.add_argument("--password", default=DEVICE_PASS)
    parser.add_argument("--port", type=int, default=DEVICE_PORT)
    parser.add_argument("--flight", type=int, help="Set active flight before syncing")
    parser.add_argument("--wait", type=int, default=3, help="Seconds to wait after flag sync")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Run sync immediately over SSH instead of waiting for the running app",
    )
    args = parser.parse_args()

    print("Connecting to {}@{}:{} ...".format(args.user, args.host, args.port))
    try:
        client = make_client(args.host, args.user, args.password, args.port)
    except Exception as exc:
        print("Connection failed:", exc)
        return 1

    try:
        if args.direct:
            run_direct_sync(client, args.flight)
        else:
            trigger_flag_sync(client, args.flight, args.wait)
        print_status(client, args.flight)
    except Exception as exc:
        print("Sync trigger failed:", exc)
        return 1
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
