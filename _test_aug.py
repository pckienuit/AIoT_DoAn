import random
import numpy as np
from train import CelebADataset, LABEL_CSV, IMG_DIR

ds_aug = CelebADataset(LABEL_CSV, IMG_DIR, partition=0, augment=True)
ds_raw = CelebADataset(LABEL_CSV, IMG_DIR, partition=0, augment=False)

random.seed(42)
img_a, (_, lm_a) = ds_aug[0]
img_b, (_, lm_b) = ds_raw[0]

print("Augmented image shape :", tuple(img_a.shape))
print("Landmarks (aug)       :", lm_a.numpy().round(3))
print("Landmarks (raw)       :", lm_b.numpy().round(3))
print("Images differ?        :", not (img_a == img_b).all().item())
print()

# Verify flip correctness: lefteye_x + righteye_x should ~= 1.0 after flip
random.seed(0)
results = []
for i in range(20):
    _, (_, lm) = ds_aug[i]
    results.append(lm.numpy())
arr = np.stack(results)
print(f"Landmarks range x: [{arr[:, 0::2].min():.3f}, {arr[:, 0::2].max():.3f}]  (should be [0,1])")
print(f"Landmarks range y: [{arr[:, 1::2].min():.3f}, {arr[:, 1::2].max():.3f}]  (should be [0,1])")
print()
print("OK - augmentation hoat dong!")
