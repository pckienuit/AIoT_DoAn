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
                print(f"Moved file: {f} -> {dest_dir}")
            except Exception as e:
                print(f"Error moving file {f}: {e}")

def move_dir(src, dest):
    if os.path.exists(src) and os.path.isdir(src):
        try:
            # If target dir already exists, shutil.move might nest it. Let's handle it.
            target = os.path.join(dest, os.path.basename(src))
            if os.path.exists(target):
                print(f"Directory {target} already exists, skipping move to avoid nesting.")
            else:
                shutil.move(src, target)
                print(f"Moved DIR: {src} -> {dest}")
        except Exception as e:
            print(f"Error moving DIR {src}: {e}")

if __name__ == '__main__':
    print("=== Starting Workspace Cleanup & Reorganization ===")
    
    # 1. Create clean directories
    create_dirs([
        'data',
        'models/checkpoints',
        'models/exports',
        'scripts/train',
        'scripts/tests',
        'scripts/sync',
        'scripts/utils',
        'scripts/export'
    ])
    
    # 2. Move Datasets & large data files
    move_dir('celebA_dataset', 'data')
    move_dir('CASIAWebFace_dataset', 'data')
    move_dir('images', 'data')
    move_dir('neg_debug', 'data')
    
    move_files('celebA_dataset.zip', 'data')
    move_files('labels.csv', 'data')
    move_files('_weight_map.csv', 'data')
    move_files('recognize_train.log', 'data')
    move_files('face_recognize_arcface_p3.ref_files.json', 'data')
    
    # 3. Move Models
    move_files('*.pth', 'models/checkpoints')
    move_files('*.onnx', 'models/exports')
    move_files('maixhub*.zip', 'models/exports')
    
    # 4. Move Training scripts
    train_scripts = [
        'train.py', 'train_recognize.py', 'train_v8.py', 'train_v9.py',
        'prepare_data.py'
    ]
    for ts in train_scripts:
        move_files(ts, 'scripts/train')
        
    # 5. Move Sync & upload scripts
    sync_scripts = [
        'upload_casia.py', 'upload_recognize.py', 'vps_sync.py'
    ]
    for ss in sync_scripts:
        move_files(ss, 'scripts/sync')
        
    # 6. Move Test & diagnostic scripts
    test_scripts = [
        'check_vps_paths.py', 'download_model.py', 'evaluate_models.py',
        'evaluate_recognize.py', 'inference_test.py', 'inference_test_v8.py',
        'kill_and_check.py', 'reboot_force.py', 'run_device_app.py',
        'run_foreground_main.py', 'run_hardware_test.py', 'run_live_mjpeg.py',
        'run_live_test.py', 'test_camera_doc.py', 'test_device_sync.py',
        'test_edge_integration.py', 'test_hardware_init.py',
        'test_hardware_init_isolated.py', 'test_image_doc.py',
        'test_image_methods.py', 'test_imports.py', 'test_model_load.py',
        'test_recognize_webcam.py', 'webcam_test.py', 'webcam_test_v2.py',
        'visualize.py'
    ]
    for t in test_scripts:
        move_files(t, 'scripts/tests')

    # 7. Move any legacy util/export files matching pattern
    utils = [
        '_diag*.py', '_test_aug.py', 'check_dataset_structure.py',
        'check_train_progress.py', 'check_vps.py', 'debug_negatives.py',
        'download_v6.py', 'find_pth.py', 'verify_vps.py'
    ]
    for u in utils:
        move_files(u, 'scripts/utils')
        
    exports = ['export_onnx.py', 'create_calib_data.py', 'zip_model.py', 'maixcam_main.py']
    for e in exports:
        move_files(e, 'scripts/export')

    print("=== Cleanup & Reorganization Complete ===")
