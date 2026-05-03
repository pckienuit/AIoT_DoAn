import torch
import torch.nn as nn
import torchvision.models as models

class FaceDetectMultiTaskV3(nn.Module):
    def __init__(self):
        super(FaceDetectMultiTaskV3, self).__init__()
        # Load MobileNetV2 pretrained backbone
        mobilenet = models.mobilenet_v2(pretrained=False)
        self.backbone = mobilenet.features  # output: [B, 1280, 7, 7]
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Ở phiên bản withval3 (cũ), các head chỉ là 1 layer Linear đơn giản
        self.class_head = nn.Linear(1280, 1)
        self.bbox_head = nn.Linear(1280, 4)
        self.landmark_head = nn.Linear(1280, 10)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.gap(feat).flatten(1)
        
        class_out = self.class_head(feat)
        bbox_out = self.bbox_head(feat)
        landmark_out = self.landmark_head(feat)
        
        return class_out, bbox_out, landmark_out

def export_to_onnx(pth_path, onnx_path):
    print(f"Khoi tao mo hinh kien truc V3 (MobileNetV2)...")
    model = FaceDetectMultiTaskV3()
    
    print(f"Dang nap trong so tu {pth_path}...")
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    
    # Bắt buộc: Chế độ eval() để BatchNormalization không bị lỗi khi export
    model.eval()

    # NPU SG2002 yêu cầu kích thước tensor đầu vào tĩnh (static shape)
    dummy_input = torch.randn(1, 3, 224, 224, device='cpu')

    print(f"Dang xuat mo hinh ONNX ra {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11, # Tương thích tốt nhất với TPU-MLIR
        do_constant_folding=True,
        input_names=['input'],
        output_names=['class_out', 'bbox_out', 'landmark_out']
    )
    print("✅ Xuat mo hinh thanh cong!")

if __name__ == "__main__":
    export_to_onnx('face_detect_model_withval3.pth', 'face_detect_model_v3.onnx')
