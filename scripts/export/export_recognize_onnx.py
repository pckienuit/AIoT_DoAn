import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FaceRecognizeNet(nn.Module):
    def __init__(self, embedding_size: int = 128):
        super().__init__()
        mv2 = models.mobilenet_v2(weights=None)
        self.backbone = mv2.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
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
        return F.normalize(emb, p=2, dim=1)


def export_to_onnx(pth_path, onnx_path):
    print("Khoi tao FaceRecognizeNet P3 (MobileNetV2, 128D)...")
    model = FaceRecognizeNet()

    if not os.path.exists(pth_path):
        raise FileNotFoundError(pth_path)

    print(f"Dang nap trong so tu {pth_path}...")
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    clean_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        clean_state_dict[clean_key] = value

    model.load_state_dict(clean_state_dict)
    model.eval()

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    dummy_input = torch.randn(1, 3, 112, 112, device="cpu")

    print(f"Dang xuat ONNX ra {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
    )
    print("Xuat ONNX thanh cong!")


def write_mud(mud_path):
    os.makedirs(os.path.dirname(mud_path), exist_ok=True)
    mud_content = """[basic]
type = cvimodel
model = face_recognize_arcface_p3.cvimodel

[extra]
model_type = custom
input_type = rgb
mean = 127.5, 127.5, 127.5
scale = 0.0078125, 0.0078125, 0.0078125
"""
    with open(mud_path, "w") as f:
        f.write(mud_content)
    print(f"Da tao MUD: {mud_path}")


if __name__ == "__main__":
    pth_file = "models/checkpoints/face_recognize_arcface_p3.pth"
    onnx_file = "models/exports/face_recognize_arcface_p3.onnx"
    mud_file = "models/exports/face_recognize_arcface_p3.mud"
    export_to_onnx(pth_file, onnx_file)
    write_mud(mud_file)
