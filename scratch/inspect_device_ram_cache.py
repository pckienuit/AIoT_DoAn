import paramiko
import json

HOST = "10.154.36.1"
USER = "root"
PASSWORD = "root"

def main():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(HOST, username=USER, password=PASSWORD, timeout=5)
        
        # Check active flight
        stdin, stdout, stderr = ssh.exec_command("cat /root/active_flight.txt")
        print(f"Active Flight: {stdout.read().decode().strip()}")
        
        # Read flight_999999.json
        stdin, stdout, stderr = ssh.exec_command("cat /root/cache/flight_999999.json")
        content = stdout.read().decode().strip()
        print("flight_999999.json contents:")
        if content:
            try:
                data = json.loads(content)
                print(f"  Count: {data.get('count')}")
                print(f"  Synced At: {data.get('synced_at')}")
                print(f"  Has test_all_faces: {data.get('test_all_faces')}")
                print("  Items:")
                for item in data.get("items", []):
                    p = item.get("payload", {})
                    print(f"    - {p.get('passenger_name')} (Flight: {p.get('flight_number')})")
            except Exception as e:
                print("  Failed to parse JSON:", e)
        else:
            print("  Empty file!")
            
        # Check tail of main.log
        print("\n=== tail of /root/main.log ===")
        stdin, stdout, stderr = ssh.exec_command("tail -30 /root/main.log")
        print(stdout.read().decode().strip())
        
        ssh.close()
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
