"""
upload_casia.py — Upload toàn bộ CASIA-WebFace dataset lên VPS.

Tổng: ~3.1 GB
  - casia-webface/train.rec  : 2.6 GB  (file lớn nhất)
  - casia-webface/train.idx  : 8.5 MB
  - casia-webface/train.lst  : 70 MB
  - casia-webface/property   : 14 B
  - eval/*.bin               : ~573 MB (8 files)

Tính năng:
  - Resume: nếu file đang upload dở, tiếp tục từ chỗ bị ngắt
  - Progress bar rõ ràng cho từng file
  - Bỏ qua file đã upload đủ kích thước
"""

import os
import paramiko
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

LOCAL_BASE  = r"CASIAWebFace_dataset"
REMOTE_BASE = "/root/CASIAWebFace_dataset"

# Danh sách file upload theo thứ tự (nhỏ trước để verify sớm)
FILES = [
    ("casia-webface/property",   f"{REMOTE_BASE}/casia-webface/property"),
    ("casia-webface/train.idx",  f"{REMOTE_BASE}/casia-webface/train.idx"),
    ("casia-webface/train.lst",  f"{REMOTE_BASE}/casia-webface/train.lst"),
    ("eval/lfw.bin",             f"{REMOTE_BASE}/eval/lfw.bin"),
    ("eval/cfp_fp.bin",          f"{REMOTE_BASE}/eval/cfp_fp.bin"),
    ("eval/agedb_30.bin",        f"{REMOTE_BASE}/eval/agedb_30.bin"),
    ("eval/calfw.bin",           f"{REMOTE_BASE}/eval/calfw.bin"),
    ("eval/cplfw.bin",           f"{REMOTE_BASE}/eval/cplfw.bin"),
    ("eval/cfp_ff.bin",          f"{REMOTE_BASE}/eval/cfp_ff.bin"),
    ("eval/sllfw.bin",           f"{REMOTE_BASE}/eval/sllfw.bin"),
    ("eval/talfw.bin",           f"{REMOTE_BASE}/eval/talfw.bin"),
    ("casia-webface/train.rec",  f"{REMOTE_BASE}/casia-webface/train.rec"),  # File lớn nhất — để cuối
]

CHUNK = 1024 * 1024  # 1 MB chunks


def create_ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=60)
    t = c.get_transport()
    if t:
        t.set_keepalive(15)
        t.window_size = 4 * 1024 * 1024       # 4MB window
        t.packetizer.REKEY_BYTES = 2**40       # tránh rekey trong khi upload lớn
    return c


def ensure_dirs(sftp):
    for d in [
        f"{REMOTE_BASE}",
        f"{REMOTE_BASE}/casia-webface",
        f"{REMOTE_BASE}/eval",
    ]:
        try:
            sftp.stat(d)
        except FileNotFoundError:
            sftp.mkdir(d)


def upload_with_resume(sftp, local_path, remote_path):
    local_size = os.path.getsize(local_path)
    fname = os.path.basename(local_path)

    # Kiểm tra file đã tồn tại trên remote chưa
    try:
        remote_size = sftp.stat(remote_path).st_size
    except FileNotFoundError:
        remote_size = 0

    if remote_size >= local_size:
        print(f"  ✓ Skip (đã đủ): {fname} ({local_size / 1024**2:.1f} MB)")
        return True

    # Resume từ vị trí đã upload
    offset = remote_size
    mode   = "ab" if offset > 0 else "wb"
    if offset > 0:
        print(f"  ↩ Resume từ {offset / 1024**2:.1f} MB / {local_size / 1024**2:.1f} MB: {fname}")
    else:
        print(f"  ↑ Upload: {fname} ({local_size / 1024**2:.1f} MB)")

    with tqdm(total=local_size, initial=offset, unit="B", unit_scale=True,
              unit_divisor=1024, desc=f"    {fname[:30]}") as pbar:
        with open(local_path, "rb") as lf:
            lf.seek(offset)
            with sftp.open(remote_path, mode) as rf:
                rf.set_pipelined(True)
                while True:
                    chunk = lf.read(CHUNK)
                    if not chunk:
                        break
                    rf.write(chunk)
                    pbar.update(len(chunk))

    # Verify
    final_size = sftp.stat(remote_path).st_size
    if final_size == local_size:
        print(f"  ✓ OK: {fname}")
        return True
    else:
        print(f"  ✗ Size mismatch: local={local_size}, remote={final_size}")
        return False


def main():
    print("=" * 55)
    print(" CASIA-WebFace Upload → VPS")
    print(f" Target: {USER}@{HOST}:{PORT}")
    print("=" * 55)

    # Tính tổng kích thước cần upload
    total_bytes = 0
    skip_bytes  = 0
    for rel, _ in FILES:
        lp = os.path.join(LOCAL_BASE, rel.replace("/", os.sep))
        if os.path.exists(lp):
            total_bytes += os.path.getsize(lp)

    print(f" Tổng: {total_bytes / 1024**3:.2f} GB\n")

    client = create_ssh()
    sftp   = client.open_sftp()

    print("[Step 1] Tạo thư mục remote...")
    ensure_dirs(sftp)

    print(f"\n[Step 2] Upload {len(FILES)} files...\n")
    failed = []
    for rel, remote_path in FILES:
        local_path = os.path.join(LOCAL_BASE, rel.replace("/", os.sep))
        if not os.path.exists(local_path):
            print(f"  ✗ Không tìm thấy local: {local_path}")
            failed.append(rel)
            continue
        try:
            ok = upload_with_resume(sftp, local_path, remote_path)
            if not ok:
                failed.append(rel)
        except Exception as e:
            print(f"  ✗ Lỗi khi upload {rel}: {e}")
            print("  Reconnecting...")
            try:
                sftp.close()
                client.close()
            except Exception:
                pass
            client = create_ssh()
            sftp   = client.open_sftp()
            failed.append(rel)

    sftp.close()

    print("\n" + "=" * 55)
    if not failed:
        print(" ✅ Upload hoàn tất!")
        print("\n Để bắt đầu train trên VPS:")
        print(f"   ssh -p {PORT} {USER}@{HOST}")
        print("   source ai_env/bin/activate")
        print("   nohup python3 train_recognize.py > recognize_train.log 2>&1 &")
        print("   tail -f recognize_train.log")
    else:
        print(f" ⚠ {len(failed)} file thất bại: {failed}")
        print(" Chạy lại script để resume.")
    print("=" * 55)

    client.close()


if __name__ == "__main__":
    main()
