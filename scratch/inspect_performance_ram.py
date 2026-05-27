import paramiko
import time

HOST = "10.154.36.1"
USER = "root"
PASSWORD = "root"

def main():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(HOST, username=USER, password=PASSWORD, timeout=5)
        
        print("=== DEVICE PROCESS AND MEMORY ===")
        stdin, stdout, stderr = ssh.exec_command("free -m")
        print(stdout.read().decode().strip())
        
        stdin, stdout, stderr = ssh.exec_command("ps aux | grep main.py | grep -v grep")
        print(stdout.read().decode().strip() or "main.py is NOT running!")
        
        print("\n=== tail -30 /root/main.log ===")
        stdin, stdout, stderr = ssh.exec_command("tail -30 /root/main.log")
        print(stdout.read().decode().strip())
        
        ssh.close()
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
