"""Quick NME debug script for v9 on test partition (partition=2, no negatives)."""
import sys, os
sys.path.insert(0, '.')
from evaluate_models import FaceDetectMultiTask, load_model_weights, CelebAValDataset, evaluate_model
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = FaceDetectMultiTask()
model = load_model_weights(model, 'models/checkpoints/face_detect_model_vps_finetune_v9.pth')
model = model.to(device)

# Use test partition (2) with NO negatives — pure face landmark eval
ds = CelebAValDataset('labels.csv', 'celebA_dataset/img_align_celeba/img_align_celebA',
                      partition=2, neg_ratio=0.0)
ds.data = ds.data.iloc[:500].reset_index(drop=True)
loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

print(f"Test samples: {len(ds)}")

m = evaluate_model(model, loader, device)
print(f"\n=== v9 Results (test partition, no negatives) ===")
print(f"  Classification: acc={m['cls_accuracy']:.4f}  F1={m['cls_f1']:.4f}  AUC={m['cls_auc_roc']:.4f}")
print(f"  BBox          : MSE={m['bbox_mse']:.5f}  MAE={m['bbox_mae']:.5f}")
print(f"  Landmarks     : MSE={m['landmark_mse']:.5f}  MAE={m['landmark_mae']:.5f}  NME={m['landmark_nme']:.2f}%")
print(f"  Combined Loss : {m['combined_loss']:.4f}")

# Also print raw landmark arrays to see actual scale
print(f"\n  n_positive={m['n_positive']} / n_total={m['n_samples']}")
