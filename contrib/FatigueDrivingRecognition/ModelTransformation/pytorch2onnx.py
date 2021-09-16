import onnx
import os
import argparse
from models.pfld import PFLDInference
import torch
import onnxsim

parser = argparse.ArgumentParser(description='pytorch2onnx')
parser.add_argument(
    '--torch_model',
    default="./checkpoint/v2/v2.pth")
parser.add_argument('--onnx_model', default="./pfld_106.onnx")
args = parser.parse_args()

print("=====> load pytorch checkpoint...")
plfd_backbone = PFLDInference()
plfd_backbone.eval()
plfd_backbone.load_state_dict(torch.load(args.torch_model, map_location=torch.device('cpu'))['plfd_backbone'])
print("PFLD bachbone:", plfd_backbone)

print("=====> convert pytorch model to onnx...")
dummy_input = torch.randn(1, 3, 112, 112)
input_names = ["input_1"]
output_names = ["output_1"]


torch.onnx.export(
    plfd_backbone,
    dummy_input,
    args.onnx_model,
    verbose=True,
    opset_version = 11,
    input_names=input_names,
    output_names=output_names)


print("====> check onnx model...")
model = onnx.load(args.onnx_model)
onnx.checker.check_model(model)