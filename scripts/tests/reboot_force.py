#!/usr/bin/env python3
import paramiko
import time
import sys

HOST = "10.154.35.1"
USER = "root"
PASS = "root"

def main():
    print("Attempting to send force reboot command to MaixCAM (retrying on connection errors)...")
    for i in range(1, 20):
        print(f"Try {i}/20...")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(HOST, username=USER, password=PASS, timeout=3)
            print("Connected! Sending reboot...")
            client.exec_command("reboot")
            client.close()
            print("Reboot command sent successfully!")
            return
        except Exception as e:
            print("  Connection attempt failed:", e)
            time.sleep(1)
    print("Could not connect to send reboot. Please power cycle the MaixCAM board manually (unplug and replug the USB cable).")

if __name__ == "__main__":
    main()
