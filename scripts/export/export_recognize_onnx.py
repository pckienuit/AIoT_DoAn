import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import os

class FaceRecognizeNet(nn.Module):
    def __init__(self, embedding_size: int = 128):
        super().__init__()
        mv2 = models.mobilenet_v2(weights=None)
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
        emb = self.head(x)
        # Tra ve embedding da duoc chuan hoa (L2-norm) de de dang so sanh Cosine
        return F.normalize(emb, p=2, dim=1)

def export_to_onnx(pth_path, onnx_path):
    print(f"Khoi tao mo hinh FaceRecognizeNet (MobileNetV2)...")
    model = FaceRecognizeNet()
    
    print(f"Dang nap trong so tu {pth_path}...")
    checkpoint = torch.load(pth_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        print("Phat hien state_dict nam trong 'model_state_dict'")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # Hinh anh dau vao cho Face Recognition la 112x112
    dummy_input = torch.randn(1, 3, 112, 112, device='cpu')

    print(f"Dang xuat mo hinh ONNX ra {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11, 
        do_constant_folding=True,
        input_names=['input'],
        output_names=['embedding']
    )
    print("✅ Xuat mo hinh thanh cong!")

if __name__ == "__main__":
    # Path relative to project root (d:\AIoT_DoAn)
    pth_file = 'models/checkpoints/face_recognize_arcface.pth'
    onnx_file = 'models/checkpoints/face_recognize_arcface.onnx'
    export_to_onnx(pth_file, onnx_file)
