"""
eval_all.py — Chay evaluate tren nhieu model lien tiep
"""
import subprocess
import os

models = [
    r"d:\AIoT_DoAn\face_detect_model_vps_finetune_v4.pth",
    r"d:\AIoT_DoAn\face_detect_model_vps_finetune_v5.pth",
    r"d:\AIoT_DoAn\face_detect_model_vps_finetune_v6.pth",
]

venv_python = r"d:\AIoT_DoAn\.venv\Scripts\python.exe"
eval_script = r"d:\AIoT_DoAn\evaluate.py"

for model in models:
    if not os.path.exists(model):
        print(f"[SKIP] {model} not found, skipping...")
        continue
    print(f"\n{'='*60}")
    print(f"Evaluating: {os.path.basename(model)}")
    print('='*60)
    result = subprocess.run(
        [venv_python, "-X", "utf8", eval_script, model],
        capture_output=False,
    )
    print(f"\n[Exit code: {result.returncode}]")
