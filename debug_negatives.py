import torch
import cv2
import numpy as np
import os
import random
from train import CelebADataset, LABEL_CSV, IMG_DIR, IMAGE_SIZE

def main():
    # Save to a local folder first
    save_dir = "neg_debug"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    dataset = CelebADataset(LABEL_CSV, IMG_DIR, partition=0, augment=False)
    
    print(f"Extracting negative samples from {IMG_DIR}...")
    count = 0
    idx = 0
    # Try first 500 images to find 5 negatives
    while count < 5 and idx < 500:
        if idx % 5 == 0:
            try:
                img_tensor, (label, bbox, landmarks) = dataset[idx]
                
                # Convert CHW tensor back to HWC numpy
                img_np = img_tensor.numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                save_path = os.path.join(save_dir, f"neg_{count}.png")
                cv2.imwrite(save_path, img_np)
                print(f"Saved {save_path} (label={label.item()})")
                count += 1
            except Exception as e:
                print(f"Error at idx {idx}: {e}")
        idx += 1
    
    if count == 0:
        print("No negative samples found in the first 500 indices.")

if __name__ == "__main__":
    main()
