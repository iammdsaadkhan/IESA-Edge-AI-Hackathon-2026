import torch
from train_model import SEMNet_Final_1_5M

NUM_CLASSES = 8
MODEL_PATH = "models/best_model.pth"
ONNX_PATH = "models/SEMNet_Final_1_5M.onnx"

model = SEMNet_Final_1_5M(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

dummy = torch.randn(1, 1, 160, 160)

torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    do_constant_folding=True
)

print("✅ ONNX exported:", ONNX_PATH)
