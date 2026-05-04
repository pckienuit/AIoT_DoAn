import paramiko
import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS)
    sftp = client.open_sftp()
    
    remote_cvimodel = "/root/maixcam_workspace/face_detect.cvimodel"
    local_cvimodel = os.path.join("models", "exports", "face_detect.cvimodel")
    
    try:
        sftp.stat(remote_cvimodel)
        print(f"File {remote_cvimodel} exists on VPS. Downloading...")
        sftp.get(remote_cvimodel, local_cvimodel)
        print(f"Downloaded to {local_cvimodel}")
        
        mud_content = """[basic]
type = cvimodel
model = face_detect.cvimodel

[extra]
model_type = custom
input_type = rgb
mean = 0, 0, 0
scale = 0.0039215686, 0.0039215686, 0.0039215686
"""
        local_mud = os.path.join("models", "exports", "face_detect_v3.mud")
        with open(local_mud, 'w') as f:
            f.write(mud_content)
        print("Created .mud file.")
    except Exception as e:
        print(f"File does not exist yet or error: {e}")

if __name__ == '__main__':
    main()
