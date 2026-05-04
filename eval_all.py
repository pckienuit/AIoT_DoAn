import subprocess
import os
import re
import sys

models = [
    r"d:\AIoT_DoAn\models\checkpoints\face_detect_model_vps_finetune_v3.pth",
    r"d:\AIoT_DoAn\models\checkpoints\face_detect_model_vps_finetune_v4.pth",
    r"d:\AIoT_DoAn\models\checkpoints\face_detect_model_vps_finetune_v5.pth",
    r"d:\AIoT_DoAn\models\checkpoints\face_detect_model_vps_finetune_v6.pth",
    r"d:\AIoT_DoAn\models\checkpoints\face_detect_model_vps_finetune_v7.pth",
    r"d:\AIoT_DoAn\models\checkpoints\face_detect_model_vps_finetune_v8.pth",
    r"d:\AIoT_DoAn\models\checkpoints\face_detect_model_vps_finetune_v9.pth",
]

venv_python = r"d:\AIoT_DoAn\.venv\Scripts\python.exe"
eval_script = r"d:\AIoT_DoAn\evaluate.py"

summary = []

for model in models:
    if not os.path.exists(model):
        print(f"[SKIP] {model} not found, skipping...")
        continue
        
    model_name = os.path.basename(model)
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print('='*60)
    
    # Dùng Popen để bắt output realtime và parse kết quả
    process = subprocess.Popen(
        [venv_python, "-u", "-X", "utf8", eval_script, model],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    
    acc = "N/A"
    nme_clean = "N/A"
    nme_all = "N/A"
    
    # Đọc output từng dòng để in ra màn hình và trích xuất thông số
    for line in iter(process.stdout.readline, ''):
        try:
            sys.stdout.write(line)
        except UnicodeEncodeError:
            sys.stdout.write(line.encode('ascii', 'replace').decode('ascii'))
        sys.stdout.flush()
        
        # Parse metrics
        if "Classification Accuracy:" in line:
            match = re.search(r"Classification Accuracy:\s*([\d\.]+)%", line)
            if match: acc = match.group(1)
        elif "NME (% interocular) clean:" in line:
            match = re.search(r"clean:\s*([\d\.]+)%", line)
            if match: nme_clean = match.group(1)
        elif "NME (% interocular) all+:" in line:
            match = re.search(r"all\+:\s*([\d\.]+)%", line)
            if match: nme_all = match.group(1)
                
    process.wait()
    print(f"\n[Exit code: {process.returncode}]")
    
    # Lưu lại để làm bảng tổng hợp
    short_name = model_name.replace("face_detect_model_vps_finetune_", "").replace(".pth", "")
    summary.append({
        "Model": short_name,
        "Acc (%)": acc,
        "NME Clean (%)": nme_clean,
        "NME All (%)": nme_all,
    })

# In bảng tổng hợp cuối cùng
print("\n" + "="*70)
print(f"{'FINAL EVALUATION SUMMARY':^70}")
print("="*70)
print(f"{'Model Version':<20} | {'Accuracy (%)':<14} | {'NME Clean (%)':<14} | {'NME All (%)':<14}")
print("-" * 70)
for res in summary:
    print(f"{res['Model']:<20} | {res['Acc (%)']:<14} | {res['NME Clean (%)']:<14} | {res['NME All (%)']:<14}")
print("="*70)
