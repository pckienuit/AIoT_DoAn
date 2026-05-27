import sys
import paramiko

DEVICE_HOST = "10.154.36.1"
DEVICE_USER = "root"
DEVICE_PASS = "root"

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_device_cmd.py '<command>'")
        sys.exit(1)
        
    cmd = sys.argv[1]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(DEVICE_HOST, port=22, username=DEVICE_USER, password=DEVICE_PASS, timeout=10)
        _, stdout, stderr = client.exec_command(cmd)
        out = stdout.read().decode()
        err = stderr.read().decode()
        rc = stdout.channel.recv_exit_status()
        
        print(f"--- Command: {cmd} ---")
        print(f"Exit Code: {rc}")
        if out:
            print("--- Standard Output ---")
            print(out)
        if err:
            print("--- Standard Error ---")
            print(err)
    except Exception as e:
        print("Failed to connect/execute:", e)
    finally:
        client.close()

if __name__ == "__main__":
    main()
