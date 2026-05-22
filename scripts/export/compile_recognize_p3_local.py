import os
import subprocess
import sys

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


def main():
    if not os.path.exists(LOCAL_ONNX):
        raise FileNotFoundError(
            f"Missing {LOCAL_ONNX}. Run: python scripts/export/export_recognize_onnx.py"
        )

    ensure_local_mud()

    workspace_dir = os.getcwd().replace("\\", "/")
    print(f"Workspace directory: {workspace_dir}")

    # Mount current workspace directory to /workspace inside docker
    # Output cvimodel and mlir will be placed inside models/exports/
    bash_cmd = (
        "pip install tpu_mlir[onnx] -i https://pypi.org/simple && "
        "model_transform.py "
        "--model_name face_recognize_arcface_p3 "
        "--model_def models/exports/face_recognize_arcface_p3.onnx "
        "--input_shapes [[1,3,112,112]] "
        "--mean 127.5,127.5,127.5 "
        "--scale 0.0078125,0.0078125,0.0078125 "
        "--pixel_format rgb "
        "--mlir models/exports/face_recognize_arcface_p3.mlir && "
        "model_deploy.py "
        "--mlir models/exports/face_recognize_arcface_p3.mlir "
        "--quantize F16 "
        "--processor cv181x "
        "--model models/exports/face_recognize_arcface_p3.cvimodel"
    )

    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{workspace_dir}:/workspace",
        "-w", "/workspace",
        "sophgo/tpuc_dev:latest",
        "bash", "-lc",
        bash_cmd
    ]

    print("Running Docker container to compile P3 ONNX to cvimodel...")
    print("This might take a few minutes. Please wait...")

    print(f"Command list: {docker_cmd}")

    result = subprocess.run(docker_cmd)
    if result.returncode != 0:
        print("Error compiling model!")
        sys.exit(result.returncode)

    print("Compilation completed successfully!")
    print(f"Compiled cvimodel at: {LOCAL_CVIMODEL}")
    print(f"MUD file at: {LOCAL_MUD}")


if __name__ == "__main__":
    main()
