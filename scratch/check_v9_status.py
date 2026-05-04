import paramiko
import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

def check_status():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, port=PORT, username=USER, password=PASS)
        
        # Check if process is running
        stdin, stdout, stderr = client.exec_command("ps aux | grep train_v9.py | grep -v grep")
        processes = stdout.read().decode().strip()
        
        if processes:
            print("--- Training Process Found ---")
            print(processes)
        else:
            print("--- Training Process NOT Running ---")
            
        # Look for log files
        stdin, stdout, stderr = client.exec_command("ls -lt *.log")
        logs = stdout.read().decode().strip()
        print("\n--- Recent Log Files ---")
        print(logs if logs else "No .log files found.")
        
        # If there's a log file, tail the last few lines
        # Common log names might be train_v9.log or similar
        stdin, stdout, stderr = client.exec_command("tail -n 20 train_v9.log 2>/dev/null || tail -n 20 nohup.out 2>/dev/null")
        output = stdout.read().decode().strip()
        if output:
            print("\n--- Last 20 lines of log (train_v9.log or nohup.out) ---")
            print(output)
        else:
            # Try to find any output file modified recently
            stdin, stdout, stderr = client.exec_command("ls -t | head -n 5")
            recent_files = stdout.read().decode().strip()
            print("\n--- Recently modified files on VPS ---")
            print(recent_files)
            
        client.close()
    except Exception as e:
        print(f"Error connecting to VPS: {e}")

if __name__ == "__main__":
    check_status()
