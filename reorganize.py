import os
import shutil
import glob

def create_dirs(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def move_files(pattern, dest_dir):
    files = glob.glob(pattern)
    for f in files:
        if os.path.isfile(f):
            try:
                shutil.move(f, os.path.join(dest_dir, os.path.basename(f)))
                print(f"Moved {f} -> {dest_dir}")
            except Exception as e:
                print(f"Error moving {f}: {e}")

def move_dir(src, dest):
    if os.path.exists(src) and os.path.isdir(src):
        try:
            shutil.move(src, os.path.join(dest, os.path.basename(src)))
            print(f"Moved DIR {src} -> {dest}")
        except Exception as e:
            print(f"Error moving {src}: {e}")

if __name__ == '__main__':
    print("Bat dau don dep du an...")
    
    # 1. Tạo các thư mục
    create_dirs([
        'data',
        'models/checkpoints',
        'models/exports',
        'scripts/utils',
        'scripts/export'
    ])
    
    # 2. Di chuyển Data
    move_dir('celebA_dataset', 'data')
    move_dir('images', 'data')
    move_dir('neg_debug', 'data')
    move_files('celebA_dataset.zip', 'data')
    move_files('labels.csv', 'data')
    
    # 3. Di chuyển Models
    move_files('*.pth', 'models/checkpoints')
    move_files('*.onnx', 'models/exports')
    move_files('maixhub*.zip', 'models/exports')
    
    # 4. Di chuyển Utils / Scripts tạm
    utils = [
        '_diag*.py', '_test_aug.py', 'check_dataset_structure.py',
        'check_train_progress.py', 'check_vps.py', 'debug_negatives.py',
        'download_v6.py', 'find_pth.py', 'verify_vps.py', 'vps_sync.py'
    ]
    for u in utils:
        move_files(u, 'scripts/utils')
        
    # 5. Di chuyển Export scripts
    exports = ['export_onnx.py', 'create_calib_data.py', 'zip_model.py', 'maixcam_main.py']
    for e in exports:
        move_files(e, 'scripts/export')

    print("Don dep thu muc hoan tat!")
