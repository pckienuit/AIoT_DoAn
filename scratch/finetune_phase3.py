"""
finetune_phase3.py — ArcFace Phase 3: Enhanced Augmentation Warm Restart

Strategy:
  - Load weights from Phase 2 (face_recognize_arcface_ft.pth)
  - Reset optimizer + scheduler (warm restart)
  - LR: 0.003 -> 1e-5 (CosineAnnealing, 30 epochs)
  - Enhanced augmentation: rotation, blur, perspective, erasing
  - ArcFace: S=64, M=0.50 (keep stable)
  - Output: face_recognize_arcface_p3.pth

Expected: 94.82% -> 95.0%+ LFW
"""

import os
import math
import struct
import io
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import pickle

# ==========================================
# CONFIG
# ==========================================
DATASET_DIR    = "CASIAWebFace_dataset/casia-webface"
EVAL_DIR       = "CASIAWebFace_dataset/eval"
RESUME_PATH    = "models/checkpoints/face_recognize_arcface_ft.pth"   # Phase 2 output
MODEL_OUT      = "models/checkpoints/face_recognize_arcface_p3.pth"   # Phase 3 output
EMBEDDING_SIZE = 128
BATCH_SIZE     = 128
EPOCHS         = 30
LR             = 0.003     # Phase 3: 0.003 (safer than Phase2's 0.005)
WEIGHT_DECAY   = 5e-4
ARC_S          = 64.0
ARC_M          = 0.50      # Keep stable (don't increase margin yet)
GRAD_CLIP      = 5.0
EARLY_STOP_PATIENCE = 8

# Kaggle override
_KAGGLE = "/kaggle/input"
if os.path.exists(_KAGGLE):
    for kd in os.listdir(_KAGGLE):
        candidate = os.path.join(_KAGGLE, kd, "casia-webface")
        if os.path.exists(candidate):
            DATASET_DIR = candidate
    MODEL_OUT = "/kaggle/working/face_recognize_arcface_p3.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================
# 1. RecordIO Dataset
# ==========================================
class RecordIODataset(Dataset):
    def __init__(self, rec_path, idx_path, transform=None, lst_path=None):
        self.rec_path  = rec_path
        self.transform = transform
        self.keys      = []
        self.offsets   = {}
        with open(idx_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    key    = int(parts[0])
                    offset = int(parts[1])
                    self.keys.append(key)
                    self.offsets[key] = offset
        self._rec_file = None

    def _open(self):
        if self._rec_file is None:
            self._rec_file = open(self.rec_path, "rb")

    def _read_record(self, offset):
        self._open()
        self._rec_file.seek(offset)
        buf = self._rec_file.read(8)
        if len(buf) < 8:
            raise IOError(f"Truncated record at offset {offset}")
        magic, length_flag = struct.unpack("=II", buf)
        length = length_flag & ((1 << 29) - 1)
        data   = self._rec_file.read(length)
        if len(data) < 26:
            raise IOError(f"Record too short at offset {offset}")
        flag, label, idx_id = struct.unpack("IfQ", data[:16])
        if flag != 0:
            raise IOError(f"Not an image record (flag={flag})")
        img_bytes = data[24:]
        if len(img_bytes) < 2 or img_bytes[0] != 0xFF or img_bytes[1] != 0xD8:
            raise IOError(f"Invalid JPEG at offset {offset}")
        return img_bytes, int(label)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        for attempt in range(10):
            try:
                key    = self.keys[(idx + attempt) % len(self.keys)]
                offset = self.offsets[key]
                img_bytes, label = self._read_record(offset)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label
            except Exception as e:
                if attempt == 0 and "Not an image record" not in str(e):
                    print(f"[Dataset] Skip bad record idx={idx}: {e}")
        return torch.zeros(3, 112, 112), -1

    def __del__(self):
        if self._rec_file:
            try:
                self._rec_file.close()
            except Exception:
                pass


# ==========================================
# 2. ArcFace Loss
# ==========================================
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, num_classes, s=64.0, m=0.50):
        super().__init__()
        self.s      = s
        self.m      = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        num_classes = self.weight.size(0)
        bad = (labels < 0) | (labels >= num_classes)
        if bad.any():
            labels = labels.clamp(0, num_classes - 1)
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        sine   = torch.sqrt(1.0 - cosine ** 2)
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.s
        return F.cross_entropy(output, labels)


# ==========================================
# 3. Backbone
# ==========================================
class FaceRecognizeNet(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        mv2 = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.backbone = mv2.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

    def get_embedding(self, x):
        return F.normalize(self.forward(x), p=2, dim=1)


# ==========================================
# 4. LFW Evaluator
# ==========================================
def load_bin(path, image_size=(112, 112)):
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


def evaluate_lfw(model, pairs, device):
    model.eval()
    dists, labels = [], []
    with torch.no_grad():
        for img1, img2, same in pairs:
            e1 = model.get_embedding(img1.unsqueeze(0).to(device))
            e2 = model.get_embedding(img2.unsqueeze(0).to(device))
            dist = 1.0 - F.cosine_similarity(e1, e2).item()
            dists.append(dist)
            labels.append(same)
    best_acc, best_th = 0.0, 0.0
    max_dist = max(dists) if dists else 1.0
    for th in np.arange(0.001, max_dist + 0.01, 0.001):
        preds = [1 if d < th else 0 for d in dists]
        acc = sum(p == lb for p, lb in zip(preds, labels)) / len(labels)
        if acc > best_acc:
            best_acc = acc
            best_th  = th
    print(f"     [LFW eval] Best threshold={best_th:.3f} | Acc={best_acc*100:.2f}%")
    return best_acc


# ==========================================
# 5. Training
# ==========================================
def get_num_classes(dataset_dir):
    prop = os.path.join(dataset_dir, "property")
    with open(prop, "r") as f:
        return int(f.read().strip().split(",")[0])


def train():
    rec_path = os.path.join(DATASET_DIR, "train.rec")
    idx_path = os.path.join(DATASET_DIR, "train.idx")
    if not os.path.exists(rec_path):
        print(f"ERROR: {rec_path} not found!")
        return

    num_classes = get_num_classes(DATASET_DIR)
    print(f"CASIA-WebFace: {num_classes} identities | Embedding: {EMBEDDING_SIZE}D")

    # ── Enhanced Augmentation (Phase 3 key change) ──────────────────────
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # Standard color jitter (same as before)
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
        # NEW: Random rotation ±10 degrees (head tilt variation)
        transforms.RandomRotation(degrees=10),
        # NEW: Random perspective (camera angle variation)
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        # NEW: Gaussian blur (focus/lens variation)
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
        # Grayscale (lighting fallback)
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # NEW: Random erasing (occlusion robustness: glasses, mask, hair)
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
    ])
    # ────────────────────────────────────────────────────────────────────

    lst_path = os.path.join(DATASET_DIR, "train.lst")
    dataset  = RecordIODataset(rec_path, idx_path, transform=transform_train, lst_path=lst_path)

    def safe_collate(batch):
        batch = [(img, lbl) for img, lbl in batch if lbl >= 0]
        if not batch:
            return None, None
        return torch.utils.data.dataloader.default_collate(batch)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=safe_collate,
        persistent_workers=True,
        drop_last=True,
    )
    print(f"   Dataset size: {len(dataset):,} images")
    print(f"   Augmentations: flip, jitter, rotation±10, perspective, blur, erasing")

    model   = FaceRecognizeNet(embedding_size=EMBEDDING_SIZE).to(device)
    arcface = ArcFaceLoss(in_features=EMBEDDING_SIZE, num_classes=num_classes,
                          s=ARC_S, m=ARC_M).to(device)

    if torch.cuda.device_count() > 1:
        print(f"[Multi-GPU] Using {torch.cuda.device_count()} GPUs")
        model   = nn.DataParallel(model)
        arcface = nn.DataParallel(arcface)

    optimizer = optim.SGD(
        [{"params": model.parameters()},
         {"params": arcface.parameters()}],
        lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # ── Phase 3 Resume: load weights only, reset optimizer ──────────────
    start_epoch      = 0
    best_lfw         = 0.0
    no_improve_count = 0

    if os.path.exists(RESUME_PATH):
        print(f"\nPhase 3: Loading weights from Phase 2: {RESUME_PATH}")
        ckpt = torch.load(RESUME_PATH, map_location=device, weights_only=False)
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        raw_arc = arcface.module if isinstance(arcface, nn.DataParallel) else arcface
        if "arcface_state_dict" in ckpt:
            raw_arc.load_state_dict(ckpt["arcface_state_dict"])
        # DO NOT load optimizer/scheduler — warm restart
        best_lfw = ckpt.get("lfw_acc", 0.0)
        print(f"   Loaded weights | Phase2 LFW={best_lfw*100:.2f}% | Reset LR={LR:.3e}")
    else:
        print(f"ERROR: Resume path not found: {RESUME_PATH}")
        return
    # ────────────────────────────────────────────────────────────────────

    lfw_pairs = None
    lfw_bin   = os.path.join(EVAL_DIR, "lfw.bin")
    if os.path.exists(lfw_bin):
        print("Loading LFW eval...")
        lfw_pairs = load_bin(lfw_bin)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    print(f"\nPhase 3 | Epochs=1..{EPOCHS} | LR={LR} | S={ARC_S} | M={ARC_M} | EarlyStop={EARLY_STOP_PATIENCE}")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        arcface.train()
        total_loss  = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for images, labels in pbar:
            if images is None:
                continue
            images = images.to(device)
            labels = labels.to(device).long()

            valid_mask = labels >= 0
            if not valid_mask.any():
                continue
            images = images[valid_mask]
            labels = labels[valid_mask]

            if images.size(0) < 2:
                continue

            optimizer.zero_grad()
            embeddings = model(images)
            loss       = arcface(embeddings, labels)

            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            all_params = list(model.parameters()) + list(arcface.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
            optimizer.step()

            total_loss  += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        lr_now   = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | LR: {lr_now:.2e} | Batches: {num_batches}")

        if lfw_pairs and (epoch + 1) % 5 == 0:
            eval_model = model.module if isinstance(model, nn.DataParallel) else model
            lfw_acc    = evaluate_lfw(eval_model, lfw_pairs, device)
            print(f"  --> LFW Accuracy: {lfw_acc * 100:.2f}%")

            if lfw_acc > best_lfw:
                best_lfw         = lfw_acc
                no_improve_count = 0
                eval_arc = arcface.module if isinstance(arcface, nn.DataParallel) else arcface
                torch.save({
                    "model_state_dict":     eval_model.state_dict(),
                    "arcface_state_dict":   eval_arc.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch":                epoch + 1,
                    "lfw_acc":              lfw_acc,
                    "embedding_size":       EMBEDDING_SIZE,
                    "phase":                3,
                    "augmentation":         "rotation,perspective,blur,erasing",
                }, MODEL_OUT)
                print(f"  --> Saved best model | LFW: {best_lfw*100:.2f}%")
            else:
                no_improve_count += 1
                print(f"  --> No improvement ({no_improve_count}/{EARLY_STOP_PATIENCE}) | Best: {best_lfw*100:.2f}%")
                if no_improve_count >= EARLY_STOP_PATIENCE:
                    print(f"\nEarly stopping at Epoch {epoch+1}. Best LFW: {best_lfw*100:.2f}%")
                    break

    print(f"\nPhase 3 Done! Model: {MODEL_OUT}")
    if lfw_pairs:
        print(f"   Best LFW Accuracy: {best_lfw * 100:.2f}%")


if __name__ == "__main__":
    train()
