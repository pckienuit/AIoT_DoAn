"""
restart_train.py - Kill train cũ, xoá checkpoint lỗi, khởi động train mới từ đầu.
"""
import os
import paramiko
import time
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

REMOTE_CHECKPOINT = "/root/models/checkpoints/face_recognize_arcface.pth"

def run(client, cmd, label=""):
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if label:
        print(f"\n[{label}]")
    if out:
        print(out)
    if err and "no process" not in err.lower() and err.strip():
        print(f"  [!] {err}")
    return out

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

    # 1. Kill train cũ
    run(client, "pkill -f train_recognize.py || true", "Kill process cu")
    time.sleep(2)

    # 2. Xoá checkpoint cũ (không có arcface weights -> sẽ gây lỗi resume)
    run(client,
        f"rm -f {REMOTE_CHECKPOINT} && echo 'Da xoa checkpoint cu'",
        "Xoa checkpoint loi")

    # 3. Kích hoạt train mới hoàn toàn từ epoch 1
    cmd_train = (
        "nohup /root/ai_env/bin/python3 /root/train_recognize.py "
        "> /root/recognize_train.log 2>&1 &"
    )
    run(client, cmd_train, "Khoi dong train moi (epoch 1)")
    time.sleep(4)

    # 4. Xác nhận process đang chạy
    out = run(client, "pgrep -a -f train_recognize.py", "Kiem tra process")
    if out:
        print("OK - Process dang chay!")
    else:
        print("CANH BAO: Khong thay process, kiem tra log!")

    # 5. Hiển thị log ban đầu
    run(client, "tail -20 /root/recognize_train.log", "Log khoi dong")

    client.close()

if __name__ == "__main__":
    main()
