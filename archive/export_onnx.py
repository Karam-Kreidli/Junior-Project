import torch
import torch.nn as nn
from bioreef.models.backbone import ViTBackbone
from bioreef.models.mceam import MCEAM
import sys
import io

# Force UTF-8 for PyTorch's internal ONNX exporter logs on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class BioReefStage1ONNXWrapper(nn.Module):
    def __init__(self, backbone, mceam, head):
        super().__init__()
        self.backbone = backbone
        self.mceam = mceam
        self.head = head

    def forward(self, roi, social, habitat, full_frame):
        streams = {"roi": roi, "social": social, "habitat": habitat, "full_frame": full_frame}
        features = self.backbone(streams)
        out = self.mceam(features)
        return self.head(out['embedding'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = ViTBackbone(freeze=True).to(device)
mceam = MCEAM(embed_dim=768, num_context_levels=3, output_dim=256, num_heads=8).to(device)
head = nn.Linear(256, 3).to(device)

ckpt = torch.load("bioreef_stage1_final.pt", weights_only=True, map_location=device)
mceam.load_state_dict(ckpt['mceam'])
head.load_state_dict(ckpt['head'])

wrapper = BioReefStage1ONNXWrapper(backbone, mceam, head).to(device)
wrapper.eval()

d1 = torch.randn(1, 3, 224, 224).to(device)
d2 = torch.randn(1, 3, 224, 224).to(device)
d3 = torch.randn(1, 3, 224, 224).to(device)
d4 = torch.randn(1, 3, 224, 224).to(device)

torch.onnx.export(
    wrapper,
    (d1, d2, d3, d4),
    "bioreef_v1.onnx",
    export_params=True,
    opset_version=18,
    input_names=['roi', 'social', 'habitat', 'full'],
    output_names=['logits']
)
print("ONNX EXPORT SUCCESSFUL")
