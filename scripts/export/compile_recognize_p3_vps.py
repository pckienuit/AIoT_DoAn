import os
import time

import paramiko
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("VPS_HOST")
PORT = int(os.getenv("VPS_PORT", 22))
USER = os.getenv("VPS_USER")
PASS = os.getenv("VPS_PASS")

REMOTE_DIR = "/root/maixcam_recognize_p3"
LOCAL_ONNX = os.path.join("models", "exports", "face_recognize_arcface_p3.onnx")
LOCAL_CVIMODEL = os.path.join("models", "exports", "face_recognize_arcface_p3.cvimodel")
LOCAL_MUD = os.path.join("models", "exports", "face_recognize_arcface_p3.mud")


def ensure_local_mud():
    os.makedirs(os.path.dirname(LOCAL_MUD), exist_ok=True)
    mud_content = """[basic]
type = cvimodel
model = face_recognize_arcface_p3.cvimodel

[extra]
model_type = custom
input_type = rgb
mean = 127.5, 127.5, 127.5
scale = 0.0078125, 0.0078125, 0.0078125
"""
    with open(LOCAL_MUD, "w") as f:
        f.write(mud_content)


def stream_channel(stdout, stderr):
    while not stdout.channel.exit_status_ready():
        while stdout.channel.recv_ready():
            print(stdout.channel.recv(4096).decode("utf-8", "ignore"), end="")
        while stderr.channel.recv_stderr_ready():
            print(stderr.channel.recv_stderr(4096).decode("utf-8", "ignore"), end="")
        time.sleep(0.2)

    while stdout.channel.recv_ready():
        print(stdout.channel.recv(4096).decode("utf-8", "ignore"), end="")
    while stderr.channel.recv_stderr_ready():
        print(stderr.channel.recv_stderr(4096).decode("utf-8", "ignore"), end="")
    return stdout.channel.recv_exit_status()


def main():
    if not os.path.exists(LOCAL_ONNX):
        raise FileNotFoundError(
            f"Missing {LOCAL_ONNX}. Run: python scripts/export/export_recognize_onnx.py"
        )

    ensure_local_mud()

    print(f"Connecting to VPS {HOST}:{PORT}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS)
    sftp = client.open_sftp()

    client.exec_command(f"rm -rf {REMOTE_DIR} && mkdir -p {REMOTE_DIR}/images")
    remote_onnx = f"{REMOTE_DIR}/face_recognize_arcface_p3.onnx"
    print("Uploading P3 ONNX...")
    sftp.put(LOCAL_ONNX, remote_onnx)

    docker_script = f"""
set -e
cd {REMOTE_DIR}
if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sh
fi
docker pull sophgo/tpuc_dev:latest
docker run --rm -v $(pwd):/workspace/model_convert -w /workspace/model_convert sophgo/tpuc_dev:latest bash -lc "
    pip install tpu_mlir[onnx] -i https://pypi.org/simple
    model_transform.py \\
        --model_name face_recognize_arcface_p3 \\
        --model_def face_recognize_arcface_p3.onnx \\
        --input_shapes [[1,3,112,112]] \\
        --mean 127.5,127.5,127.5 \\
        --scale 0.0078125,0.0078125,0.0078125 \\
        --pixel_format rgb \\
        --mlir face_recognize_arcface_p3.mlir
    model_deploy.py \\
        --mlir face_recognize_arcface_p3.mlir \\
        --quantize F16 \\
        --processor cv181x \\
        --model face_recognize_arcface_p3.cvimodel
"
"""

    print("Compiling P3 on VPS...")
    _, stdout, stderr = client.exec_command(docker_script, get_pty=True)
    status = stream_channel(stdout, stderr)
    if status != 0:
        raise RuntimeError("P3 compile failed on VPS")

    print("Downloading compiled cvimodel...")
    sftp.get(f"{REMOTE_DIR}/face_recognize_arcface_p3.cvimodel", LOCAL_CVIMODEL)
    print(f"Saved: {LOCAL_CVIMODEL}")
    print(f"Saved: {LOCAL_MUD}")

    sftp.close()
    client.close()


if __name__ == "__main__":
    main()
