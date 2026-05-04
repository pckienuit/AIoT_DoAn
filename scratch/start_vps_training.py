import paramiko
import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

def start_training():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, port=PORT, username=USER, password=PASS)
        
        # Start the training process in the background using nohup
        # Redirecting both stdout and stderr to train_v9.log
        command = "source ai_env/bin/activate && nohup python3 train_v9.py > train_v9.log 2>&1 &"
        print(f"Executing: {command}")
        
        stdin, stdout, stderr = client.exec_command(command)
        
        # Give it a second to start and write to log
        import time
        time.sleep(2)
        
        # Check if process is running
        stdin, stdout, stderr = client.exec_command("ps aux | grep train_v9.py | grep -v grep")
        processes = stdout.read().decode().strip()
        
        if processes:
            print("\n--- Training Process Started Successfully ---")
            print(processes)
            
            # Show first few lines of log
            stdin, stdout, stderr = client.exec_command("head -n 20 train_v9.log")
            print("\n--- Log Head ---")
            print(stdout.read().decode().strip())
        else:
            print("\n--- Failed to start training process ---")
            stdin, stdout, stderr = client.exec_command("cat train_v9.log")
            print(stdout.read().decode().strip())
            
        client.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    start_training()
