"""
train_recognize.py — Face Recognition với CASIA-WebFace (MXNet RecordIO format)

Pipeline:
    CASIA-WebFace .rec/.idx -> MobileNetV2 Backbone -> Embedding 128D -> ArcFace Loss

Dataset: CASIAWebFace_dataset/casia-webface/
  - train.rec   : ảnh dạng RecordIO (đã crop 112x112)
  - train.idx   : index byte offset của từng record
  - property    : "10572,112,112" (num_classes, H, W)

Eval: CASIAWebFace_dataset/eval/lfw.bin, cfp_fp.bin, agedb_30.bin
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
MODEL_OUT      = "models/checkpoints/face_recognize_arcface.pth"
EMBEDDING_SIZE = 128      # 128D cho nhẹ, 512D cho chính xác cao hơn
BATCH_SIZE     = 128
EPOCHS         = 60       # Tăng lên 60 để model hội tụ sâu hơn
LR             = 0.05    # SGD dùng LR 0.05 cho Batch=128
WEIGHT_DECAY   = 5e-4
ARC_S          = 64.0     # ArcFace scale (thường 64 với dataset lớn)
ARC_M          = 0.50     # ArcFace margin (góc tăng thêm)
GRAD_CLIP      = 5.0
RESUME         = True     # True = tiếp tục train từ checkpoint cũ nếu có
EARLY_STOP_PATIENCE = 8   # Dừng sớm nếu LFW không cải thiện sau N lần eval

# Kaggle override
_KAGGLE = "/kaggle/input"
if os.path.exists(_KAGGLE):
    for kd in os.listdir(_KAGGLE):
        candidate = os.path.join(_KAGGLE, kd, "casia-webface")
        if os.path.exists(candidate):
            DATASET_DIR = candidate
    MODEL_OUT = "/kaggle/working/face_recognize_arcface.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. MXNet RecordIO Reader (không cần MXNet)
# ==========================================
# InsightFace RecordIO format:
# [offset+0]  magic(4) + length_flag(4)   — 8 bytes  (xử lý khi read)
# [offset+8]  header data (16 bytes): int32 label tại byte 0-3, padding còn lại
# [offset+24] JPEG data bắt đầu tại đây
INSIGHTFACE_HEADER_SIZE = 24  # 8 (rec prefix) + 16 (internal header)

class RecordIODataset(Dataset):
    """
    Đọc trực tiếp file .rec và .idx của MXNet RecordIO format.
    Không cần cài MXNet.
    """
    def __init__(self, rec_path: str, idx_path: str, transform=None,
                 lst_path: str = None):
        self.rec_path  = rec_path
        self.transform = transform

        # Đọc index file để biết byte offset của từng record
        self.keys = []
        self.offsets = {}
        with open(idx_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    key    = int(parts[0])
                    offset = int(parts[1])
                    self.keys.append(key)
                    self.offsets[key] = offset

        self._rec_file = None  # Mở lazy để multiprocessing worker tự mở


    def _open(self):
        if self._rec_file is None:
            self._rec_file = open(self.rec_path, "rb")

    def _read_record(self, offset: int):
        """
        Đọc 1 record theo InsightFace RecordIO format:
          [0..8)  : magic(4) + length_flag(4)
          [8..24) : 16-byte internal header: flag(I), label(f), id(Q)
          [24..)  : JPEG image data
        """
        self._open()
        self._rec_file.seek(offset)

        # 8 bytes prefix: magic + length_flag
        buf = self._rec_file.read(8)
        if len(buf) < 8:
            raise IOError(f"Truncated record at offset {offset}")
        magic, length_flag = struct.unpack("=II", buf)
        length = length_flag & ((1 << 29) - 1)  # data length

        # Đọc toàn bộ data
        data = self._rec_file.read(length)

        if len(data) < 26:
            raise IOError(f"Record too short ({len(data)} bytes) at offset {offset}")

        # Parse internal header (16 bytes)
        header = data[:16]
        flag, label, idx_id = struct.unpack("IfQ", header)
        
        # flag = 2 thường là các identity/list records ở cuối file, không phải ảnh
        if flag != 0:
            raise IOError(f"Not an image record (flag={flag})")

        # JPEG bắt đầu tại byte 24 của data
        img_bytes = data[24:]

        # Validate JPEG magic bytes (FF D8)
        if len(img_bytes) < 2 or img_bytes[0] != 0xFF or img_bytes[1] != 0xD8:
            raise IOError(f"Invalid JPEG at offset {offset}: first bytes={img_bytes[:4].hex()}")

        return img_bytes, int(label)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # Retry tối đa 10 record khác nếu gặp ảnh hỏng hoặc list record
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
                # Chỉ log ở attempt đầu tiên để tránh rác console
                if attempt == 0 and "Not an image record" not in str(e):
                    print(f"[Dataset] Skip bad record idx={idx} (key={key}): {e}")
        # Fallback: trả về ảnh đen với label -1 (được bỏ trong training loop)
        dummy = torch.zeros(3, 112, 112)
        return dummy, -1

    def __del__(self):
        if self._rec_file:
            try:
                self._rec_file.close()
            except Exception:
                pass


# ==========================================
# 2. ARC-FACE LOSS
# ==========================================
class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Face Recognition.
    Ref: Deng et al., CVPR 2019
    """
    def __init__(self, in_features: int, num_classes: int,
                 s: float = 64.0, m: float = 0.50):
        super().__init__()
        self.s   = s
        self.m   = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m  = math.cos(m)
        self.sin_m  = math.sin(m)
        self.th     = math.cos(math.pi - m)
        self.mm     = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        num_classes = self.weight.size(0)

        # Safety clamp: đảm bảo label nằm trong [0, num_classes-1]
        # (phòng trường hợp label bị corrupt từ dataset)
        bad = (labels < 0) | (labels >= num_classes)
        if bad.any():
            print(f"[ArcFace] WARNING: {bad.sum().item()} label(s) out of range "
                  f"[0,{num_classes}): {labels[bad].tolist()[:5]}. Clamping.")
            labels = labels.clamp(0, num_classes - 1)

        # L2-normalize embeddings và prototype weights
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        sine = torch.sqrt(1.0 - cosine ** 2)
        phi  = cosine * self.cos_m - sine * self.sin_m   # cos(θ + m)
        phi  = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.s

        return F.cross_entropy(output, labels)


# ==========================================
# 3. BACKBONE — MobileNetV2 (nhẹ cho MaixCAM)
# ==========================================
class FaceRecognizeNet(nn.Module):
    def __init__(self, embedding_size: int = 128):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Inference only: trả về unit-norm embedding."""
        emb = self.forward(x)
        return F.normalize(emb, p=2, dim=1)


# ==========================================
# 4. LFW / CFP / AgeDB EVALUATOR
# ==========================================
def load_bin(path: str, image_size=(112, 112)):
    """Load .bin eval file (LFW format) dùng pickle."""
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


def evaluate_lfw(model: nn.Module, pairs, device):
    """Tinh Accuracy tren LFW bang Cosine Similarity voi best-threshold sweep.

    Khong dung threshold cung -- tim threshold toi uu tren chinh tap pairs,
    giong cach chuan trong benchmark InsightFace / DeepFace.
    """
    model.eval()
    dists  = []
    labels = []

    with torch.no_grad():
        for img1, img2, same in pairs:
            e1 = model.get_embedding(img1.unsqueeze(0).to(device))
            e2 = model.get_embedding(img2.unsqueeze(0).to(device))
            dist = 1.0 - F.cosine_similarity(e1, e2).item()
            dists.append(dist)
            labels.append(same)

    # Sweep threshold de tim best accuracy (khong co tien gian dinh)
    best_acc = 0.0
    best_th  = 0.0
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
# 5. TRAINING
# ==========================================
def get_num_classes(dataset_dir: str) -> int:
    prop = os.path.join(dataset_dir, "property")
    with open(prop, "r") as f:
        num_classes = int(f.read().strip().split(",")[0])
    return num_classes


def train():
    rec_path = os.path.join(DATASET_DIR, "train.rec")
    idx_path = os.path.join(DATASET_DIR, "train.idx")

    if not os.path.exists(rec_path):
        print(f"❌ Không tìm thấy: {rec_path}")
        return

    num_classes = get_num_classes(DATASET_DIR)
    print(f"✅ CASIA-WebFace: {num_classes} identities | Embedding: {EMBEDDING_SIZE}D")

    # Augmentation — ảnh gốc đã là 112x112
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    lst_path   = os.path.join(DATASET_DIR, "train.lst")
    dataset    = RecordIODataset(rec_path, idx_path, transform=transform_train,
                                 lst_path=lst_path)

    def safe_collate(batch):
        """Bỏ qua các sample có label=-1 (dummy fallback) trước khi tạo tensor."""
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
        drop_last=True,  # tránh batch cuối ít sample gây lỗi BatchNorm
    )
    print(f"   Dataset size: {len(dataset):,} ảnh")

    # Model + Loss
    model   = FaceRecognizeNet(embedding_size=EMBEDDING_SIZE).to(device)
    arcface = ArcFaceLoss(
        in_features=EMBEDDING_SIZE,
        num_classes=num_classes,
        s=ARC_S, m=ARC_M,
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"[Multi-GPU] Sử dụng {torch.cuda.device_count()} GPUs")
        model   = nn.DataParallel(model)
        arcface = nn.DataParallel(arcface)

    # Sử dụng SGD với Momentum (Chuẩn cho ArcFace) thay vì AdamW (gây Mode Collapse)
    optimizer = optim.SGD(
        [{"params": model.parameters()},
         {"params": arcface.parameters()}],
        lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # ── Resume từ checkpoint cũ nếu có ──────────────────────────────────
    start_epoch      = 0
    best_lfw         = 0.0
    no_improve_count = 0

    if RESUME and os.path.exists(MODEL_OUT):
        print(f"\n🔄 Resume từ checkpoint: {MODEL_OUT}")
        ckpt = torch.load(MODEL_OUT, map_location=device, weights_only=False)

        # 1. Load backbone weights
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        raw_model.load_state_dict(ckpt["model_state_dict"])

        # 2. Load ArcFace weight (quan trọng! nếu bỏ sẽ re-init ngẫu nhiên → Loss vọật lên)
        raw_arc = arcface.module if isinstance(arcface, nn.DataParallel) else arcface
        if "arcface_state_dict" in ckpt:
            raw_arc.load_state_dict(ckpt["arcface_state_dict"])

        # 3. Load optimizer + scheduler để LR và momentum buffer không bị reset
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # 4. Khôi phục metadata
        start_epoch = ckpt.get("epoch", 0)
        best_lfw    = ckpt.get("lfw_acc", 0.0)
        print(f"   ✅ Đã load epoch={start_epoch} | Best LFW={best_lfw*100:.2f}% | LR hiện tại={optimizer.param_groups[0]['lr']:.2e}")
    else:
        print(f"\n🆕 Bắt đầu train mới từ đầu")
    # ────────────────────────────────────────────────────────────────────

    # Load LFW eval
    lfw_pairs = None
    lfw_bin   = os.path.join(EVAL_DIR, "lfw.bin")
    if os.path.exists(lfw_bin):
        print("✅ Tải LFW eval...")
        lfw_pairs = load_bin(lfw_bin)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    print(f"\n🚀 Train ArcFace | Epochs={start_epoch+1}..{EPOCHS} | LR={LR} | S={ARC_S} | M={ARC_M} | EarlyStop patience={EARLY_STOP_PATIENCE}")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        arcface.train()
        total_loss  = 0.0
        num_batches = 0  # đếm batch thực tế (tránh chia sai khi có None batch)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for images, labels in pbar:
            if images is None:  # batch hoàn toàn dummy, bỏ qua
                continue
            images = images.to(device)
            labels = labels.to(device).long()

            # Filter nốt nếu có label âm lọt qua collate
            valid_mask = labels >= 0
            if not valid_mask.any():
                continue
            images = images[valid_mask]
            labels = labels[valid_mask]

            # BatchNorm cần >= 2 samples
            if images.size(0) < 2:
                continue

            optimizer.zero_grad()
            embeddings = model(images)
            loss       = arcface(embeddings, labels)

            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            # Clip gradient CẢ model VÀ arcface (arcface weight cũng có grad)
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

        # LFW Evaluation mỗi 5 epoch
        if lfw_pairs and (epoch + 1) % 5 == 0:
            eval_model = model.module if isinstance(model, nn.DataParallel) else model
            lfw_acc = evaluate_lfw(eval_model, lfw_pairs, device)
            print(f"  --> LFW Accuracy: {lfw_acc * 100:.2f}%")

            if lfw_acc > best_lfw:
                best_lfw = lfw_acc
                no_improve_count = 0
                # Lưu đủ 5 key: model + arcface + optimizer + scheduler + metadata
                eval_arc = arcface.module if isinstance(arcface, nn.DataParallel) else arcface
                torch.save({
                    "model_state_dict":     eval_model.state_dict(),
                    "arcface_state_dict":   eval_arc.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch":                epoch + 1,
                    "lfw_acc":              lfw_acc,
                    "embedding_size":       EMBEDDING_SIZE,
                }, MODEL_OUT)
                print(f"  --> Saved best model | LFW: {best_lfw*100:.2f}%")
            else:
                no_improve_count += 1
                print(f"  --> Khong cai thien ({no_improve_count}/{EARLY_STOP_PATIENCE}) | Best LFW: {best_lfw*100:.2f}%")
                if no_improve_count >= EARLY_STOP_PATIENCE:
                    print(f"\n⏹ Early stopping! LFW khong tang sau {EARLY_STOP_PATIENCE} lan eval. Dung tai Epoch {epoch+1}.")
                    break

        elif not lfw_pairs:
            # Lưu mỗi epoch nếu không có eval
            eval_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({"model_state_dict": eval_model.state_dict(),
                        "epoch": epoch + 1,
                        "embedding_size": EMBEDDING_SIZE},
                       MODEL_OUT)

    print(f"\n✅ Training xong! Model: {MODEL_OUT}")
    if lfw_pairs:
        print(f"   Best LFW Accuracy: {best_lfw * 100:.2f}%")


if __name__ == "__main__":
    train()
