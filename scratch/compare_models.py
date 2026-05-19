"""
compare_models.py - Deep evaluation & comparison of Phase 1 vs Phase 2 ArcFace models.

Metrics:
  1. LFW Accuracy (threshold sweep)
  2. Cosine Distance distribution (positive vs negative pairs)
  3. EER (Equal Error Rate)
  4. TAR@FAR=0.1%, TAR@FAR=1% (industry standard)
  5. Per-threshold precision/recall
"""
import os, sys, io, pickle, time
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from train_recognize import FaceRecognizeNet

MODELS = {
    "Phase1 (94.43%)": "models/checkpoints/face_recognize_arcface.pth",
    "Phase2 FT (94.82%)": "models/checkpoints/face_recognize_arcface_ft.pth",
}

EVAL_DIR = "CASIAWebFace_dataset/eval"
EMBEDDING_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ─── Load eval data ─────────────────────────────────────
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

# ─── Compute all embeddings ─────────────────────────────
def compute_distances(model, pairs, device):
    model.eval()
    dists, labels = [], []
    with torch.no_grad():
        for img1, img2, same in pairs:
            e1 = model.get_embedding(img1.unsqueeze(0).to(device))
            e2 = model.get_embedding(img2.unsqueeze(0).to(device))
            d = 1.0 - F.cosine_similarity(e1, e2).item()
            dists.append(d)
            labels.append(same)
    return np.array(dists), np.array(labels)

# ─── Metrics ─────────────────────────────────────────────
def compute_metrics(dists, labels):
    results = {}

    # 1. Best accuracy (threshold sweep)
    best_acc, best_th = 0.0, 0.0
    max_d = max(dists) if len(dists) > 0 else 1.0
    for th in np.arange(0.001, max_d + 0.01, 0.001):
        preds = (dists < th).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_th = th
    results['best_acc'] = best_acc
    results['best_th'] = best_th

    # 2. Distribution stats
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_dists = dists[pos_mask]
    neg_dists = dists[neg_mask]

    results['pos_mean'] = pos_dists.mean()
    results['pos_std'] = pos_dists.std()
    results['pos_median'] = np.median(pos_dists)
    results['neg_mean'] = neg_dists.mean()
    results['neg_std'] = neg_dists.std()
    results['neg_median'] = np.median(neg_dists)

    # 3. EER (Equal Error Rate)
    thresholds = np.arange(0.001, max_d + 0.001, 0.001)
    fars, frrs = [], []
    for th in thresholds:
        # FAR = False Accept Rate: negative pairs predicted as same
        far = ((dists[neg_mask] < th).sum()) / neg_mask.sum() if neg_mask.sum() > 0 else 0
        # FRR = False Reject Rate: positive pairs predicted as different
        frr = ((dists[pos_mask] >= th).sum()) / pos_mask.sum() if pos_mask.sum() > 0 else 0
        fars.append(far)
        frrs.append(frr)
    fars, frrs = np.array(fars), np.array(frrs)

    # Find EER point
    eer_idx = np.argmin(np.abs(fars - frrs))
    results['eer'] = (fars[eer_idx] + frrs[eer_idx]) / 2
    results['eer_th'] = thresholds[eer_idx]

    # 4. TAR@FAR=0.1% and TAR@FAR=1%
    for target_far in [0.001, 0.01]:
        valid = np.where(fars <= target_far)[0]
        if len(valid) > 0:
            idx = valid[-1]  # largest threshold where FAR <= target
            tar = 1.0 - frrs[idx]
        else:
            tar = 0.0
        results[f'tar@far={target_far}'] = tar

    return results

# ─── Main ─────────────────────────────────────────────────
def main():
    lfw_bin = os.path.join(EVAL_DIR, "lfw.bin")
    if not os.path.exists(lfw_bin):
        print(f"ERROR: {lfw_bin} not found!")
        return

    print("Loading LFW eval pairs...")
    pairs = load_bin(lfw_bin)
    print(f"  Loaded {len(pairs)} pairs\n")

    all_results = {}

    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"SKIP {name}: {path} not found")
            continue

        print(f"{'='*60}")
        print(f"  Evaluating: {name}")
        print(f"  Path: {path}")
        print(f"{'='*60}")

        # Load model
        model = FaceRecognizeNet(embedding_size=EMBEDDING_SIZE).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        ckpt_epoch = ckpt.get("epoch", "?")
        ckpt_lfw = ckpt.get("lfw_acc", "?")
        print(f"  Checkpoint epoch: {ckpt_epoch}")
        print(f"  Checkpoint LFW: {ckpt_lfw}")
        print()

        # Compute
        t0 = time.time()
        dists, labels = compute_distances(model, pairs, device)
        elapsed = time.time() - t0
        print(f"  Inference time: {elapsed:.1f}s ({len(pairs)/elapsed:.0f} pairs/sec)")

        metrics = compute_metrics(dists, labels)
        all_results[name] = metrics

        print(f"\n  --- Results ---")
        print(f"  LFW Accuracy:     {metrics['best_acc']*100:.2f}% (th={metrics['best_th']:.4f})")
        print(f"  EER:              {metrics['eer']*100:.2f}% (th={metrics['eer_th']:.4f})")
        print(f"  TAR@FAR=0.1%:     {metrics['tar@far=0.001']*100:.2f}%")
        print(f"  TAR@FAR=1.0%:     {metrics['tar@far=0.01']*100:.2f}%")
        print(f"\n  Positive pairs (same person):")
        print(f"    Mean dist:    {metrics['pos_mean']:.4f}")
        print(f"    Std dist:     {metrics['pos_std']:.4f}")
        print(f"    Median dist:  {metrics['pos_median']:.4f}")
        print(f"  Negative pairs (diff person):")
        print(f"    Mean dist:    {metrics['neg_mean']:.4f}")
        print(f"    Std dist:     {metrics['neg_std']:.4f}")
        print(f"    Median dist:  {metrics['neg_median']:.4f}")
        print(f"  Separability:   {metrics['neg_mean'] - metrics['pos_mean']:.4f} (neg_mean - pos_mean)")
        print()

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ─── Comparison ─────────────────────────────────────────
    if len(all_results) >= 2:
        names = list(all_results.keys())
        m1, m2 = all_results[names[0]], all_results[names[1]]

        print("="*60)
        print("  COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Metric':<25} {'Phase1':>12} {'Phase2 FT':>12} {'Delta':>10}")
        print("-"*60)

        for key, label in [
            ('best_acc', 'LFW Accuracy'),
            ('eer', 'EER'),
            ('tar@far=0.001', 'TAR@FAR=0.1%'),
            ('tar@far=0.01', 'TAR@FAR=1.0%'),
            ('pos_mean', 'Pos Mean Dist'),
            ('neg_mean', 'Neg Mean Dist'),
        ]:
            v1 = m1[key]
            v2 = m2[key]
            delta = v2 - v1
            if 'acc' in key or 'tar' in key:
                print(f"  {label:<23} {v1*100:>10.2f}% {v2*100:>10.2f}% {delta*100:>+9.2f}%")
            elif 'eer' in key:
                print(f"  {label:<23} {v1*100:>10.2f}% {v2*100:>10.2f}% {delta*100:>+9.2f}%")
            else:
                print(f"  {label:<23} {v1:>11.4f} {v2:>11.4f} {delta:>+10.4f}")

        print()
        sep1 = m1['neg_mean'] - m1['pos_mean']
        sep2 = m2['neg_mean'] - m2['pos_mean']
        print(f"  {'Separability':<23} {sep1:>11.4f} {sep2:>11.4f} {sep2-sep1:>+10.4f}")
        print()

        # Recommendation
        if m2['best_acc'] > m1['best_acc']:
            print("  VERDICT: Phase 2 model is BETTER. Use face_recognize_arcface_ft.pth")
        elif m2['best_acc'] == m1['best_acc']:
            print("  VERDICT: Models are EQUAL in accuracy. Check other metrics.")
        else:
            print("  VERDICT: Phase 1 model is still BETTER. Keep face_recognize_arcface.pth")

    print("\nDone!")

if __name__ == "__main__":
    main()
