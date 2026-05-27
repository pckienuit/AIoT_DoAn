#!/usr/bin/env python3
"""Trigger a manual cache sync on the MaixCAM device."""
import argparse
import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

import paramiko


DEVICE_HOST = "10.154.36.1"
DEVICE_USER = "root"
DEVICE_PASS = "root"
DEVICE_PORT = 22
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_ROOT / "server" / "data" / "prototype.db"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "MaixCAM_App" / "config.json"
TEST_ALL_FLIGHT_ID = 999999


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


def load_default_server_url() -> str:
    if DEFAULT_CONFIG_PATH.exists():
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return str(cfg.get("server_url", "http://127.0.0.1:8000")).rstrip("/")
    return "http://127.0.0.1:8000"


def discover_registered_flight_ids(db_path: Path) -> list[int]:
    if not db_path.exists():
        raise RuntimeError("Database not found: {}".format(db_path))

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT flight_id
            FROM bookings
            WHERE face_registered = 1
               OR qdrant_point_id IS NOT NULL
            ORDER BY flight_id
            """
        ).fetchall()

    return [int(row[0]) for row in rows if row[0] is not None]


def fetch_sync_payload(server_url: str, flight_id: int, timeout: int) -> dict:
    url = "{}/api/sync/{}".format(server_url.rstrip("/"), int(flight_id))
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def upload_test_cache(
    client: paramiko.SSHClient,
    test_flight_id: int,
    items: list[dict],
    flight_status: str = "scheduled",
) -> None:
    cache_entry = {
        "flight_id": int(test_flight_id),
        "synced_at": time.time(),
        "count": len(items),
        "items": items,
        "test_all_faces": True,
        "status": flight_status,
        "recognition_allowed": flight_status not in ("departed", "arrived", "cancelled"),
    }

    sftp = client.open_sftp()
    try:
        try:
            sftp.mkdir("/root/cache")
        except OSError:
            pass

        remote_path = "/root/cache/flight_{}.json".format(int(test_flight_id))
        with sftp.file(remote_path, "w") as remote_file:
            remote_file.write(json.dumps(cache_entry))
    finally:
        sftp.close()


def sync_all_faces_to_test_cache(
    client: paramiko.SSHClient,
    server_url: str,
    db_path: Path,
    test_flight_id: int,
    timeout: int,
) -> None:
    flight_ids = discover_registered_flight_ids(db_path)
    if not flight_ids:
        raise RuntimeError("No registered face flights found in {}".format(db_path))

    print("[all] Registered face flight IDs:", flight_ids)
    all_items = []
    combined_status = "scheduled"

    for flight_id in flight_ids:
        print("[all] Fetching flight {} from server...".format(flight_id))
        payload = fetch_sync_payload(server_url, flight_id, timeout)
        items = payload.get("items", [])
        print("[all]   {} embeddings".format(len(items)))
        all_items.extend(items)

        # Propagate departed/arrived/cancelled to test cache so recognition is blocked
        status = payload.get("status", "scheduled")
        if status in ("departed", "arrived", "cancelled"):
            combined_status = status

    upload_test_cache(client, test_flight_id, all_items, combined_status)
    rc, out = ssh_run(client, "echo {} > /root/active_flight.txt".format(int(test_flight_id)))
    if rc != 0:
        raise RuntimeError(out or "Failed to set test active flight")

    print("[all] Uploaded {} embeddings to /root/cache/flight_{}.json".format(
        len(all_items),
        int(test_flight_id),
    ))
    print("[all] Active flight set to test cache {} (status: {})".format(
        int(test_flight_id), combined_status,
    ))


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
        rc, out = ssh_run(client, "echo {} > /root/active_flight.txt".format(int(flight_id)))
        if rc != 0:
            raise RuntimeError("Failed to set active flight: {}".format(out))
        print("[direct] Active flight set to {}".format(flight_id))
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
        "--all",
        action="store_true",
        help="Sync every registered face into one test cache on the edge",
    )
    parser.add_argument(
        "--test-flight",
        type=int,
        default=TEST_ALL_FLIGHT_ID,
        help="Synthetic active flight ID used by --all",
    )
    parser.add_argument(
        "--server-url",
        default=load_default_server_url(),
        help="FastAPI server URL used by --all",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite DB used by --all to discover registered faces",
    )
    parser.add_argument(
        "--api-timeout",
        type=int,
        default=10,
        help="HTTP timeout for --all server requests",
    )
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
        if args.all:
            sync_all_faces_to_test_cache(
                client,
                args.server_url,
                args.db,
                args.test_flight,
                args.api_timeout,
            )
            print_status(client, args.test_flight)
        elif args.direct:
            run_direct_sync(client, args.flight)
            print_status(client, args.flight)
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
