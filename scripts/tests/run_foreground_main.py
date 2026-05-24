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

    print("\nRunning 'python3 -u main.py' in foreground with 10 second timeout...")
    stdin, stdout, stderr = client.exec_command("killall -9 python3; killall -9 maixvision_server; cd /root && python3 -u main.py", timeout=10)
    
    try:
        # Read from channel until it blocks/timeouts or exits
        out = []
        while not stdout.channel.exit_status_ready():
            if stdout.channel.recv_ready():
                chunk = stdout.channel.recv(4096).decode('utf-8', errors='ignore')
                sys.stdout.write(chunk)
                sys.stdout.flush()
            if stdout.channel.recv_stderr_ready():
                chunk = stdout.channel.recv_stderr(4096).decode('utf-8', errors='ignore')
                sys.stderr.write(chunk)
                sys.stderr.flush()
    except socket.timeout:
        print("\nCommand timed out (device app is still running!)")
    except Exception as e:
        print(f"\nExecution error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    main()
