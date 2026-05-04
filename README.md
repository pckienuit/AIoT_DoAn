# Face Detection & Landmark — AIoT Project

A project for training a model to perform **face detection + 5 facial landmark regression** (left eye, right eye, nose, left mouth corner, right mouth corner) on the CelebA dataset, targeting deployment on **MaixCAM** (Sipeed Edge AI Camera) — an IoT edge device.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Model & Architecture](#model--architecture)
4. [Tools & Usage](#tools--usage)
5. [.env Template](#env-template)
6. [Evaluation Results](#evaluation-results)
7. [References](#references)

---

## Project Structure

```
d-AIoT-DoAn/
├── .env                          # Environment variables (VPS, MaixCAM credentials)
├── .gitignore
├── labels.csv                    # Merged CelebA metadata: partition, bbox, landmarks, attrs
│
├── train.py                      # Original training script
├── train_v8.py                   # Training v8 (light augmentation, Wing + Focal loss)
├── train_v9.py                   # Training v9 — continuation of v8, 90 epochs, Gaussian noise, label smoothing
│
├── webcam_test.py                # Webcam: Haar Cascade (detect) → Model (landmark)
├── webcam_test_v2.py            # Webcam: Model-only 2-pass (grid scan → aligned crop)
│
├── inference_test.py             # Quick inference test on CelebA images
├── evaluate_models.py            # Comparative evaluation of all checkpoints (Acc, F1, AUC, NME, MAE)
├── prepare_data.py              # Merge 4 CelebA metadata files → labels.csv
├── reorganize.py                # Reorganize project directory structure
├── visualize.py                # Visualize training results
├── check_vps_paths.py          # Verify paths on VPS
├── vps_sync.py                 # Sync files between local and VPS via SSH/SFTP
│
├── data/
│   ├── images/                  # Calibration images for MaixHub converter
│   └── celebA_dataset/           # CelebA dataset (metadata CSVs only, raw images stored separately)
│
├── models/
│   ├── checkpoints/             # .pth checkpoints (13 versions)
│   │     ├── face_detect_model.pth
│   │     ├── face_detect_model_withval*.pth          (v1–v13)
│   │     └── face_detect_model_vps_finetune*.pth     (v1–v9)
│   └── exports/                 # Deployable files: .onnx, .mud, .cvimodel, .zip
│         ├── face_detect_v9.onnx
│         ├── face_detect_v9.mud
│         ├── face_detect_v9.cvimodel
│         └── maixhub_upload_v9.zip
│
├── scripts/
│   ├── export/
│   │   ├── export_onnx.py       # Export PyTorch → ONNX (V3 architecture)
│   │   ├── export_v9.py        # v9-specific export
│   │   ├── compile_vps.py      # Compile model (MaixHub/YOLO)
│   │   ├── create_calib_data.py  # Generate calibration images (resize to 224×224, save as JPG)
│   │   ├── zip_model.py        # Create ZIP with correct structure for MaixHub (explicit dir entry)
│   │   ├── upload_to_maixcam.py  # Upload model + script to MaixCAM via SSH/SFTP
│   │   └── maixcam_main.py    # Inference app for MaixCAM device
│   └── utils/
│         ├── check_vps.py         # Check training results on VPS
│         ├── check_train_progress.py  # Monitor training progress
│         ├── vps_sync.py         # File sync via SSH/SFTP (download .pth from VPS)
│         ├── verify_vps.py        # Validate checkpoint integrity
│         ├── debug_negatives.py  # Debug hard negative samples
│         ├── find_pth.py        # Find the latest .pth file
│         └── download_v6.py     # Download checkpoint to local
│
├── MaixCAM_App/
│   └── main.py                  # Inference app for MaixCAM (2-stage: YOLO detect → landmark)
│
├── docs/
│   ├── webcam_tracking_pipeline_v2.md  # Model-only webcam pipeline documentation
│   ├── maixhub_zip_issue.md           # Bug report: ZIP structure for MaixHub
│   └── crop_padding_bug.md            # Bug report: zero-padding vs BORDER_REPLICATE
│
└── results/
    ├── evaluation_results.csv    # Evaluation results for all models
    └── eval_v9_fixed.csv        # v9 evaluation results
```

---

## Installation

### Requirements

- Python 3.10+
- PyTorch (CUDA if GPU available)
- OpenCV (`opencv-python`)
- pandas, numpy, scikit-learn
- `paramiko` (for SSH/SFTP to VPS and MaixCAM)
- `python-dotenv` (reads `.env` file)
- `tqdm` (progress bars)
- `matplotlib` (optional, for evaluation plots)

```bash
pip install torch torchvision opencv-python pandas numpy scikit-learn paramiko python-dotenv tqdm matplotlib
```

### Dataset

CelebA dataset should be placed at `data/celebA_dataset/` with the following structure:

```
data/celebA_dataset/
├── list_eval_partition.csv        # Train/val/test split
├── list_bbox_celebA.csv          # Bounding box annotations
├── list_landmarks_align_celebA.csv  # 5 facial landmarks
├── list_attr_celebA.csv          # 40 binary attributes
└── (raw images img_align_celebA/)  # Stored separately, metadata only needed here
```

Run `prepare_data.py` to merge the 4 metadata files into `labels.csv`:

```bash
python prepare_data.py
```

---

## Model & Architecture

### Backbone

**MobileNetV2** (pre-trained on ImageNet) with 3 prediction heads:

| Head | Output | Shape | Activation |
|------|--------|-------|------------|
| `class_head` | Face / No-face score | `(B, 1)` | Sigmoid |
| `bbox_head` | Bounding box `[x, y, w, h]` | `(B, 4)` | Clamp [0,1] |
| `landmark_head` | 5 landmarks × 2 coords | `(B, 10)` | Clamp [0,1] |

### Loss Functions

```
Total Loss = CE(class) + SmoothL1(bbox) + LM_LOSS_WEIGHT × (Wing + Focal)
```

- **Wing Loss** (Feng et al., CVPR 2018): emphasizes small errors → better landmark precision
- **Focal Landmark Loss**: focuses on hard samples (occlusion, extreme pose)
- **BCE with Label Smoothing** (v9): classification regularization
- **Gaussian Noise** on landmark targets (v9): additional regularization

### Training Versions

| Version | Description | Loss weight | LR | Epochs |
|---------|-------------|-------------|-----|--------|
| `train.py` | Original | CLS + BBOX + LM×20 | 1e-4 | 50 |
| `train_v8.py` | Light augmentation, Wing+Focal | LM×20 | 1e-5 | 60 |
| `train_v9.py` | Continuation of v8, Gaussian noise, label smoothing | LM×30 | 5e-6 | 90 |

---

## Tools & Usage

### Training

```bash
# Train v9 (recommended)
python train_v9.py

# Train v8
python train_v8.py

# Original training
python train.py
```

### Webcam Test

```bash
# Method 1: Haar Cascade + Model landmark (simple, stable)
python webcam_test.py

# Method 2: Model-only 2-pass pipeline (no Haar, heavier GPU usage)
python webcam_test_v2.py
```

### Model Evaluation

```bash
# Evaluate all checkpoints (Acc, F1, AUC, NME, MAE)
python evaluate_models.py

# Limit sample count for faster evaluation
python evaluate_models.py --max_samples 5000
```

### Quick Inference Test

```bash
# Test inference on CelebA images
python inference_test.py
```

### Export Model to ONNX

```bash
# Export V3 architecture (MobileNetV2 + 3 heads)
python scripts/export/export_onnx.py

# Or call directly in Python:
from export_onnx import export_to_onnx
export_to_onnx(
    pth_path="models/checkpoints/face_detect_model_vps_finetune_v9.pth",
    onnx_path="models/exports/face_detect_v9.onnx"
)
```

### Generate Calibration Images for MaixHub

```bash
# Generate 100 calibration images (resize to 224×224)
python scripts/export/create_calib_data.py

# Customize sample count by editing the file or calling:
from create_calib_data import create_calibration_dataset
create_calibration_dataset(
    csv_file="labels.csv",
    img_dir="data/celebA_dataset/img_align_celebA/img_align_celebA",
    output_dir="data/images",
    num_samples=100
)
```

### Create ZIP for MaixHub

```bash
# Create ZIP with correct structure (explicit directory entry for MaixHub)
python scripts/export/zip_model.py

# Output: models/exports/maixhub_upload_v9.zip
```

### Upload to MaixCAM

```bash
# Upload model + script to MaixCAM device via SSH/SFTP
python scripts/export/upload_to_maixcam.py

# After upload, run on the device:
# python /root/maixcam_main.py
# or:
# python /root/main.py
```

### VPS File Sync

```bash
# Run sync (reads credentials from .env)
python vps_sync.py

# Or use individual utility scripts:
python scripts/utils/check_vps.py              # Check results on VPS
python scripts/utils/vps_sync.py               # Sync: download checkpoints
python scripts/utils/check_train_progress.py   # Monitor training progress
python scripts/utils/verify_vps.py             # Validate checkpoint
```

### Data Preparation & Organization

```bash
# Merge CelebA metadata → labels.csv
python prepare_data.py

# Reorganize project directory (move scattered files to proper locations)
python reorganize.py

# Verify paths on VPS
python check_vps_paths.py

# Visualize evaluation results
python visualize.py
```

---

## .env Template

```bash
# =============================================
# VPS Configuration (remote training)
# =============================================
VPS_HOST=your_vps_ip_address
VPS_PORT=22
VPS_USER=root
VPS_PASS=your_vps_password

# =============================================
# MaixCAM Configuration (deploy to IoT device)
# =============================================
# Default MaixCAM device credentials
MAIXCAM_HOST=10.154.36.1
MAIXCAM_PORT=22
MAIXCAM_USER=root
MAIXCAM_PASS=root

# MaixCAM2 credentials (if using a different device)
# MAIXCAM_HOST=10.154.36.2
# MAIXCAM_PORT=22
# MAIXCAM_USER=root
# MAIXCAM_PASS=root
```

---

## Evaluation Results

See `results/evaluation_results.csv` and `results/eval_v9_fixed.csv` for detailed results.

Key metrics:

- **Classification**: Accuracy, Precision, Recall, F1, AUC-ROC
- **Bounding Box**: MSE, MAE
- **Landmarks**: MSE, MAE, **NME** (Normalized Mean Error — standard landmark metric)
- **Combined Loss**: aggregated loss score

Latest checkpoint: `face_detect_model_vps_finetune_v9.pth` (fine-tuned from v8, 90 epochs)

---

## References

### Internal Docs (`docs/`)

- **`webcam_tracking_pipeline_v2.md`** — Details of the model-only 2-pass webcam pipeline: grid scan, active tracking, EMA smoothing, anti-jitter, anti-drift, false-positive rejection
- **`maixhub_zip_issue.md`** — Bug report: why ZIP created on Windows fails to recognize the `images/` directory on MaixHub Linux
- **`crop_padding_bug.md`** — Bug report: zero-padding causes NME explosion → fix with `BORDER_REPLICATE`

### Dataset

- **CelebA** (Liu et al., 2015): [https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

### Papers

- **Wing Loss** (Feng et al., CVPR 2018): emphasizes small errors for better landmark precision
- **Focal Loss** (Lin et al., ICCV 2017): focuses on hard samples

### Hardware & Tools

- **Sipeed MaixCAM**: [https://wiki.sipeed.com/maixpy](https://wiki.sipeed.com/maixpy)
- **MaixHub Model Converter**: [https://maixhub.com](https://maixhub.com)
