import paramiko
import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

def check_log_epochs():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, port=PORT, username=USER, password=PASS)
        
        # Check for epochs
        print("--- Last 10 Epoch Progress Lines ---")
        stdin, stdout, stderr = client.exec_command("grep -E 'Epoch|Saved best' train_v9.log | tail -n 20")
        print(stdout.read().decode())
        
        # Check for error again with more context
        print("\n--- Last 50 lines of log ---")
        stdin, stdout, stderr = client.exec_command("tail -n 50 train_v9.log")
        print(stdout.read().decode())
        
        client.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_log_epochs()
