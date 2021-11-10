import argparse
import torch
import os
import sys
import onnx
from utils.activations import Hardswish

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import models

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str,  
                                default='./YOLOv5_DOTAv1.5_OBB.pt', 
                                help='weights path')
parser.add_argument('--img-size', nargs='+', type=int, default=[1024, 1024], help='image size')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
opt = parser.parse_args()
opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand

# Input
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img = torch.zeros(size=(opt.batch_size, 3, *opt.img_size), device=device)

# Load PyTorch model
model = torch.load(opt.weights, map_location=device)['model'].float()
model.eval()
model.model[-1].export = True  # set Detect() layer export=True
y = model(img)  # try run


print('\nStarting ONNX export with onnx %s...' % onnx.__version__)

f = './YOLOv5_DOTAv1.5_OBB_1024_1024.onnx'

# Update model
for k, m in model.named_modules():
    m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
    if isinstance(m, models.common.Conv):
        # print(m)
        m.act = Hardswish()  # assign activation


model.fuse()  # only for ONNX
torch.onnx.export(model, img, f, export_params=True, opset_version=11, input_names=['images'],
                    output_names=['output'])

onnx_model = onnx.load(f)  # load onnx model
onnx.checker.check_model(onnx_model) # check onnx model
print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
print('ONNX export success, saved as %s' % f)