import paramiko
import os
import time
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('VPS_HOST')
PORT = int(os.getenv('VPS_PORT', 22))
USER = os.getenv('VPS_USER')
PASS = os.getenv('VPS_PASS')

def main():
    print(f"Bat dau ket noi toi VPS {HOST}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS)
    print("Ket noi SSH thanh cong!")

    sftp = client.open_sftp()
    
    # 1. Tạo thư mục làm việc trên VPS
    remote_dir = "/root/maixcam_workspace"
    client.exec_command(f"rm -rf {remote_dir} && mkdir -p {remote_dir}")
    
    # 2. Upload file maixhub_upload_fixed.zip
    local_zip = os.path.join("models", "exports", "maixhub_upload_fixed.zip")
    remote_zip = f"{remote_dir}/upload.zip"
    
    if not os.path.exists(local_zip):
        print(f"Khong tim thay file {local_zip}")
        return
        
    print("Dang tai file ZIP len VPS (co the mat 1-2 phut)...")
    sftp.put(local_zip, remote_zip)
    print("Upload hoan tat!")

    # 3. Chạy lệnh giải nén và Convert qua Docker
    docker_script = r"""
    export DEBIAN_FRONTEND=noninteractive
    cd /root/maixcam_workspace
    unzip -o upload.zip
    
    if ! command -v docker &> /dev/null
    then
        echo "Sua loi APT (Xoa script postinst cua openssh-server bi loi)..."
        rm -f /var/lib/dpkg/info/openssh-server.postinst
        dpkg --configure -a || true
        apt-get update -qq || true
        apt-get -f install -y || true
        
        echo "VPS chua cai Docker. Dang tu dong cai dat Docker (mat vai phut)..."
        curl -fsSL https://get.docker.com | sh
    fi
    
    echo "Keo Docker image cua Sophgo (neu chua co)..."
    docker pull sophgo/tpuc_dev:latest
    
    echo "Chay Container Bien dich cho SG2002..."
    docker run --rm -v $(pwd):/workspace/model_convert -w /workspace/model_convert sophgo/tpuc_dev:latest bash -c "
        echo 'Cai dat goi tpu_mlir vao container...'
        pip install tpu_mlir[onnx] -i https://pypi.org/simple
        
        echo '[Buoc 1/3] ONNX -> MLIR'
        model_transform.py \
            --model_name face_detect \
            --model_def face_detect_model_v3.onnx \
            --input_shapes [[1,3,224,224]] \
            --mean 0.0,0.0,0.0 \
            --scale 0.0039215686,0.0039215686,0.0039215686 \
            --pixel_format rgb \
            --test_input images/\$(ls images | head -n 1) \
            --test_result face_detect_top_outputs.npz \
            --mlir face_detect.mlir
            
        echo '[Buoc 2/3] Luong tu hoa (Calibration)'
        run_calibration.py face_detect.mlir \
            --dataset images \
            --input_num 100 \
            -o face_detect_calib_table
            
        echo '[Buoc 3/3] MLIR -> CVIMODEL (Danh cho mach SG2002)'
        model_deploy.py \
            --mlir face_detect.mlir \
            --quantize INT8 \
            --calibration_table face_detect_calib_table \
            --processor cv181x \
            --test_input face_detect_in_f32.npz \
            --test_reference face_detect_top_outputs.npz \
            --model face_detect.cvimodel
            
        echo 'HOAN THANH BIEN DICH!'
    "
    """
    
    print("Dang kich hoat Docker TPU-MLIR tren VPS va bien dich mo hinh...")
    print("Qua trinh nay se mat khoang 2-5 phut (tuy toc do VPS). Vui long khong tat terminal...")
    
    stdin, stdout, stderr = client.exec_command(docker_script, get_pty=True)
    
    # In log trực tiếp từ VPS về máy
    exit_status = stdout.channel.recv_exit_status()
    for line in stdout:
        clean_line = line.encode('ascii', 'ignore').decode('ascii').strip('\n')
        print(clean_line)
    for line in stderr:
        clean_line = line.encode('ascii', 'ignore').decode('ascii').strip('\n')
        print("Loi:", clean_line)

    if exit_status == 0:
        # 4. Tải file kết quả về máy
        remote_cvimodel = f"{remote_dir}/face_detect.cvimodel"
        local_cvimodel = os.path.join("models", "exports", "face_detect.cvimodel")
        print(f"Dang tai mo hinh da bien dich ({remote_cvimodel}) ve may...")
        try:
            sftp.get(remote_cvimodel, local_cvimodel)
            print("Tai ve thanh cong! File nam o:", local_cvimodel)
            
            # Tự động tạo file .mud cho MaixCAM
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
            print("Da tao file cau hinh:", local_mud)
            print("BAN DA CO THE COPY 2 FILE (.cvimodel VA .mud) VAO MAIXCAM!")
            
        except Exception as e:
            print("Loi khi tai file ve:", e)
    else:
        print("Qua trinh bien dich tren VPS that bai!")

    sftp.close()
    client.close()

if __name__ == '__main__':
    main()
