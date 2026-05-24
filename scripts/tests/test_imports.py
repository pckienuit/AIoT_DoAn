#!/usr/bin/env python3
import paramiko
import sys
import socket

HOST = "10.154.35.1"
USER = "root"
PASS = "root"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, username=USER, password=PASS, timeout=10)
        print("Connected!")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    for mod in ["nn", "camera", "display"]:
        print(f"\n--- Testing import of maix.{mod} ---")
        cmd = f"python3 -u -c \"from maix import {mod}; print('{mod} success')\""
        stdin, stdout, stderr = client.exec_command(cmd, timeout=6)
        try:
            out = stdout.read().decode().strip()
            err = stderr.read().decode().strip()
            if out:
                print("  Stdout:", out)
            if err:
                print("  Stderr:", err)
        except socket.timeout:
            print("  TIMEOUT: Command hung!")
        except Exception as e:
            print("  Error:", e)

    client.close()

if __name__ == "__main__":
    main()
