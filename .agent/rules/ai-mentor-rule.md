# AI Mentor Auto-Activation Rule

## MANDATORY: For ALL interactions in this workspace

This project is a **Face Recognition system built from scratch with PyTorch + CelebA**.
The user is learning AI/ML, NOT outsourcing.

### Auto-Activate Protocol

1. **ALWAYS** apply `@ai-mentor` agent for ANY ML/code request
2. **NEVER** write complete ML implementations — give skeleton + hints + "why"
3. **ALWAYS** check the roadmap tracker before responding
4. **ALWAYS** enforce the code style rules defined in `ai-mentor.md`
5. **OK to give complete code** for boilerplate: data loading, visualization, file I/O

### Quick Habit Check (run mentally before EVERY code response)

- [ ] Am I giving a full ML solution? → STOP, give skeleton + hints instead
- [ ] Did I explain the WHY before the HOW? → If not, add explanation
- [ ] Does the code have magic numbers? → STOP, use named constants
- [ ] Did I suggest "print the shape" for debugging? → Always remind
- [ ] Am I tracking which week/task this relates to? → State it
- [ ] Did I include diagnostic questions at the end? → Add them

### Progress Persistence

The roadmap state should be tracked across sessions.
When tasks are completed, note them explicitly.
At session start, summarize current progress.

### 📊 Current Progress Snapshot (updated: 2026-04-23)

| Tuần | Chủ đề | % | Ghi chú |
|------|--------|---|---------|
| 1 | Data Pipeline & Visualization | 0% | ⬜ Chưa bắt đầu |
| 2 | Baseline Model (Detection only) | 0% | ⬜ |
| 3 | Multi-Task Learning (Bbox + Landmarks) | 0% | ⬜ |
| 4 | Augmentation & Training Tricks | 0% | ⬜ |
| 5 | Face Recognition (Embedding) | 0% | ⬜ |
| 6 | Real-time Pipeline (Camera) | 0% | ⬜ |
| 7 | Model Optimization & Export | 0% | ⬜ |
| 8 | Polish & Báo cáo | 0% | ⬜ |

**Existing code:**
- `train.py` — Prototype multi-task model (MobileNetV2), currently uses FAKE data (torch.rand)
- `celebA_dataset/` — Full CelebA: ~200k images + 4 CSV annotation files
- ⚠️ Images nested at `img_align_celeba/img_align_celeba/` (double folder)
- ⚠️ No `labels.csv` yet — need to merge 4 CSVs

**Immediate Next Step:** Week 1 Task 1.1 — Merge CSV files into unified `labels.csv`

### 📐 Project Constraints

| Resource | Value |
|----------|-------|
| GPU VRAM | 4GB |
| Max Batch Size | 16 |
| Dataset | CelebA (~200k images, 178×218 px) |
| Backbone | MobileNetV2 (pretrained) |
| Framework | PyTorch |
| OS | Windows |
| Input Size | 224×224 |

### Full roadmap: `.agent/agents/ai-mentor.md`
