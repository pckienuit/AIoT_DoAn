import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import io
import pickle
import pandas as pd
from tqdm import tqdm

from train_recognize import FaceRecognizeNet

def load_bin(path):
    with open(path, "rb") as f:
        bins, issame = pickle.load(f, encoding="bytes")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    pairs = []
    for i in range(len(issame)):
        img1 = Image.open(io.BytesIO(bins[2 * i])).convert("RGB")
        img2 = Image.open(io.BytesIO(bins[2 * i + 1])).convert("RGB")
        pairs.append((transform(img1), transform(img2), int(issame[i])))
    return pairs

def evaluate_pairs(model, pairs, device, batch_size=128):
    model.eval()
    dists, labels = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), batch_size), desc="  Eval", leave=False):
            batch = pairs[i:i+batch_size]
            imgs1 = torch.stack([x[0] for x in batch]).to(device)
            imgs2 = torch.stack([x[1] for x in batch]).to(device)
            sames = [x[2] for x in batch]
            
            e1 = model.get_embedding(imgs1)
            e2 = model.get_embedding(imgs2)
            
            dist = 1.0 - F.cosine_similarity(e1, e2).cpu().numpy()
            dists.extend(dist.tolist())
            labels.extend(sames)
            
    best_acc, best_th = 0.0, 0.0
    max_dist = max(dists) if dists else 1.0
    for th in np.arange(0.001, max_dist + 0.01, 0.001):
        preds = [1 if d < th else 0 for d in dists]
        acc = sum(p == lb for p, lb in zip(preds, labels)) / len(labels)
        if acc > best_acc:
            best_acc = acc
            best_th  = th
    return best_acc, best_th

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    datasets = {
        "LFW (Standard)": "CASIAWebFace_dataset/eval/lfw.bin",
        "CFP-FP (Profile)": "CASIAWebFace_dataset/eval/cfp_fp.bin",
        "AgeDB-30 (Age)": "CASIAWebFace_dataset/eval/agedb_30.bin",
    }
    
    loaded_datasets = {}
    for name, path in datasets.items():
        if os.path.exists(path):
            print(f"Loading {name}...")
            loaded_datasets[name] = load_bin(path)
        else:
            print(f"Warning: {path} not found.")
            
    models_to_test = {
        "Baseline": "models/checkpoints/face_recognize_arcface.pth",
        "Phase 2": "models/checkpoints/face_recognize_arcface_ft.pth",
        "Phase 3": "models/checkpoints/face_recognize_arcface_p3.pth",
    }
    
    results = []
    
    for model_name, ckpt_path in models_to_test.items():
        if not os.path.exists(ckpt_path):
            print(f"Skipping {model_name}, not found: {ckpt_path}")
            continue
            
        print(f"\nEvaluating: {model_name}...")
        model = FaceRecognizeNet(embedding_size=128).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd)
        
        row = {"Model": model_name}
        for ds_name, pairs in loaded_datasets.items():
            acc, th = evaluate_pairs(model, pairs, device)
            print(f"  {ds_name}: Acc={acc*100:.2f}% (th={th:.3f})")
            row[f"{ds_name}"] = f"{acc*100:.2f}%"
        
        results.append(row)
        
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
