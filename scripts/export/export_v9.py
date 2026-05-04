import torch
import torch.nn as nn
import torchvision.models as models
import os

# ==========================================
# MODEL v9 Architecture
# ==========================================
class FaceDetectMultiTaskV9(nn.Module):
    def __init__(self):
        super(FaceDetectMultiTaskV9, self).__init__()
        mobilenet = models.mobilenet_v2(weights=None)
        self.backbone = mobilenet.features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.class_head    = nn.Linear(1280, 1)
        self.bbox_head    = nn.Linear(1280, 4)
        self.landmark_head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features)
        features = torch.flatten(features, 1)

        class_out    = self.class_head(features)
        bbox_out     = self.bbox_head(features)
        landmark_out = self.landmark_head(features)

        return class_out, bbox_out, landmark_out

def export_v9(pth_path, onnx_path, mud_path):
    print(f"🚀 Khoi tao mo hinh v9 (MobileNetV2)...")
    model = FaceDetectMultiTaskV9()
    
    if not os.path.exists(pth_path):
        print(f"❌ Khong tim thay file: {pth_path}")
        return

    print(f"📦 Dang nap trong so tu {pth_path}...")
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if it exists (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, device='cpu')

    print(f"🌐 Dang xuat ONNX ra {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['class_out', 'bbox_out', 'landmark_out']
    )
    print("✅ Xuat mo hinh ONNX thanh cong!")

    print(f"📝 Dang tao file .mud tai {mud_path}...")
    mud_content = f"""[basic]
type = cvimodel
model = face_detect_v9.cvimodel
input_type = rgb
mean = 0, 0, 0
std = 255, 255, 255
format = 1

[extra]
model_type = anchor_free
input_names = input
output_names = class_out, bbox_out, landmark_out
"""
    with open(mud_path, 'w') as f:
        f.write(mud_content)
    print("✅ Da tao file .mud!")

if __name__ == "__main__":
    PTH_PATH = 'face_detect_model_vps_finetune_v9.pth'
    ONNX_PATH = 'models/exports/face_detect_v9.onnx'
    MUD_PATH = 'models/exports/face_detect_v9.mud'
    
    os.makedirs('models/exports', exist_ok=True)
    export_v9(PTH_PATH, ONNX_PATH, MUD_PATH)
