import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '.')
from train_recognize import FaceRecognizeNet, load_bin

device = 'cuda'
model = FaceRecognizeNet().to(device)
ckpt = torch.load('models/checkpoints/face_recognize_arcface.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

pairs = load_bin('CASIAWebFace_dataset/eval/lfw.bin')

pos, neg = [], []
with torch.no_grad():
    for img1, img2, same in pairs:
        e1 = model.get_embedding(img1.unsqueeze(0).to(device))
        e2 = model.get_embedding(img2.unsqueeze(0).to(device))
        sim = F.cosine_similarity(e1, e2).item()
        if same == 1:
            pos.append(sim)
        else:
            neg.append(sim)

print(f"Avg pos: {np.mean(pos):.4f}")
print(f"Avg neg: {np.mean(neg):.4f}")

dists = [(1.0 - p, 1) for p in pos] + [(1.0 - n, 0) for n in neg]
best_acc = 0
best_th = 0

for th in np.arange(0.1, 1.5, 0.01):
    acc = sum((1 if d < th else 0) == s for d, s in dists) / len(dists)
    if acc > best_acc:
        best_acc = acc
        best_th = th

print(f"Best Acc: {best_acc:.4f} at Threshold: {best_th:.4f}")
