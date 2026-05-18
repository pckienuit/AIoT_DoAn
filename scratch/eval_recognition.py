import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from train_recognize import FaceRecognizeNet, load_bin, evaluate_lfw

def evaluate_downloaded_model():
    model_path = "models/checkpoints/face_recognize_arcface.pth"
    lfw_bin_path = "CASIAWebFace_dataset/eval/lfw.bin"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    if not os.path.exists(lfw_bin_path):
        print(f"Error: LFW dataset not found at {lfw_bin_path}")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device)
    embedding_size = checkpoint.get("embedding_size", 128)
    epoch = checkpoint.get("epoch", "Unknown")
    
    model = FaceRecognizeNet(embedding_size=embedding_size).to(device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print(f"Model loaded successfully (Epoch: {epoch}, Embedding Size: {embedding_size}D)")
    
    print("Loading LFW pairs...")
    lfw_pairs = load_bin(lfw_bin_path)
    print(f"Loaded {len(lfw_pairs)} pairs.")
    
    print("Evaluating...")
    accuracy = evaluate_lfw(model, lfw_pairs, device)
    
    print("=========================================")
    print(f"LFW Verification Accuracy: {accuracy * 100:.2f}%")
    print("=========================================")

if __name__ == "__main__":
    evaluate_downloaded_model()
