# AI Mentor Auto-Activation Rule

## MANDATORY: For ALL interactions in this workspace

This project is a **Face Recognition system built from scratch with PyTorch + CelebA**.
The user is learning AI/ML, NOT outsourcing.

### Auto-Activate Protocol

1. **ALWAYS** apply `@ai-mentor` agent for ANY ML/code request
2. **NEVER** write complete ML implementations — give skeleton + hints + "why"
3. **ALWAYS** check the roadmap tracker before responding
4. **ALWAYS** enforce the code style rules defined in `ai-mentor.md`
5. **OK to give complete code** for boilerplate: data loading, visualization, file I/O, webcam scripts

### Quick Habit Check (run mentally before EVERY code response)

- [ ] Am I giving a full ML solution? → STOP, give skeleton + hints instead
- [ ] Did I explain the WHY before the HOW? → If not, add explanation
- [ ] Does the code have magic numbers? → STOP, use named constants
- [ ] Did I suggest "print the shape" for debugging? → Always remind
- [ ] Am I tracking which week/task this relates to? → State it
- [ ] Did I include diagnostic questions at the end? → Add them

---

## 📊 Current Progress (updated: 2026-04-25)

| Tuần | Chủ đề | % | Ghi chú |
|------|--------|---|---------|
| **1** | Data Pipeline & Visualization | ✅ 100% | Hoàn thành |
| **2** | Baseline Model + Validation | ✅ 100% | Hoàn thành |
| **3** | Metrics (NME) | ⬜ 0% | **TIẾP THEO** |
| **4** | Augmentation & Training Tricks | ⬜ 0% | |
| **5** | Face Recognition (Embedding) | ⬜ 0% | |
| **6** | Real-time Pipeline (Camera) | 🔶 30% | Webcam chạy được, landmark chưa tốt |
| **7** | Model Optimization & Export | ⬜ 0% | |
| **8** | Polish & Báo cáo | ⬜ 0% | |

---

## ✅ Những gì đã hoàn thành

### Tuần 1 — Data Pipeline
- `prepare_data.py` — Gộp 4 file CSV → `labels.csv` (202,599 dòng, 56 cột)
- `visualize.py` — Hiển thị ảnh + landmarks (bbox từ CSV không dùng được vì tọa độ gốc)
- `labels.csv` — Train: 162,770 | Val: 19,867 | Test: ~20k (theo `list_eval_partition.csv`)
- Landmark normalized về [0,1]: `x / 178`, `y / 218`

### Tuần 2 — Training + Evaluation
- `train.py` — Full training pipeline với real CelebA data
- Validation loop sau mỗi epoch (Train loss | Val loss)
- Smoke test với 160 ảnh → pipeline OK
- Full train 10 epoch: loss hội tụ về ~0.0000
- `inference_test.py` — Inference 1 ảnh, landmark sai số <1px trên ảnh CelebA
- `webcam_test.py` — 2-stage pipeline: Haar Cascade detect → MobileNetV2 landmarks

---

## ⚠️ Known Issues (cần fix trong các tuần tiếp)

| Vấn đề | Nguyên nhân | Fix ở Tuần |
|--------|-------------|-----------|
| Classification luôn 100% | Không có negative samples (label=1 hết) | 3 hoặc 4 |
| Landmark sai trên webcam | Domain gap: Haar crop ≠ CelebA aligned format | 4 (Augmentation) + 6 (Alignment) |
| Model học "mean position" | CelebA aligned → variance nhỏ → MSE → 0 | 4 (Augmentation) |

---

## 📁 File Structure hiện tại

```
d:\AIoT_DoAn\
├── train.py                    # Model + Training loop (MobileNetV2, 3 heads)
├── prepare_data.py             # Merge 4 CSV → labels.csv
├── visualize.py                # Hiển thị ảnh + bbox + landmarks
├── inference_test.py           # Single image inference
├── webcam_test.py              # Real-time webcam 2-stage pipeline
├── labels.csv                  # 202,599 dòng × 56 cột
├── face_detect_model.pth       # Model từ lần train đầu (fake data)
├── face_detect_model_withval.pth ← Model đang dùng (train thật, 10 epochs)
└── celebA_dataset/
    ├── img_align_celeba/img_align_celeba/ ← ~202k ảnh 178×218px
    ├── list_bbox_celeba.csv
    ├── list_landmarks_align_celeba.csv
    ├── list_attr_celeba.csv
    └── list_eval_partition.csv
```

---

## 🏗️ Model Architecture

```
Input: (B, 3, 224, 224)
  └── MobileNetV2 backbone (pretrained ImageNet)
      └── AdaptiveAvgPool2d → flatten → 1280-dim
          ├── class_head    → Linear(1280, 1)   → BCEWithLogitsLoss (trivial)
          ├── bbox_head     → Linear(1280, 4)   → bị bỏ qua (dùng _)
          └── landmark_head → Linear(1280, 10)  → MSELoss ← đang dùng
```

---

## 🔧 Training Config

| Config | Giá trị |
|--------|---------|
| Backbone | MobileNetV2 (pretrained=True) |
| Optimizer | Adam, lr=0.001 |
| Batch size | 16 |
| Epochs | 10 |
| Image size | 224×224 |
| GPU | CUDA (4GB VRAM) |
| Train set | 162,770 ảnh |
| Val set | 19,867 ảnh |
| Epoch time | ~16-36 phút/epoch |

---

## 🎯 Immediate Next Step: Tuần 3 — Metrics (NME)

**NME (Normalized Mean Error)** là metric chuẩn cho facial landmark detection.

```python
# Công thức NME:
# nme = mean(euclidean_distance(pred, gt)) / normalization_factor
# normalization_factor = khoảng cách 2 mắt (inter-ocular distance)
```

Task Tuần 3:
1. Implement NME metric function
2. Evaluate model trên val/test set
3. So sánh NME trước/sau augmentation (Tuần 4)
4. Thêm negative samples để fix classification

### 📐 Project Constraints

| Resource | Giá trị |
|----------|---------|
| GPU VRAM | 4GB |
| Max Batch Size | 16 |
| Dataset | CelebA (~202k images, 178×218px) |
| Backbone | MobileNetV2 (pretrained) |
| Framework | PyTorch |
| OS | Windows |
| Python | 3.12 |

### Full agent rules: `.agent/agents/ai-mentor.md`
