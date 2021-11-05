import argparse
import os
import sys
import torch
import onnx
from models.retina import Retina 
from data import cfg_mnet

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='./weights/mnet_plate.pth', help='weights path')
opt = parser.parse_args()

cfg = cfg_mnet
model = Retina(cfg=cfg, phase='test')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_dict = torch.load(opt.weights, map_location=lambda storage, loc: storage.cuda(device))
model.load_state_dict(pretrained_dict, strict=False)
model.eval()

img = torch.randn(1, 3, 640, 640)
f = opt.weights.replace('.pth', '.onnx')  # filename
print('\nStarting ONNX export with onnx %s...' % onnx.__version__)    
torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['image']) # pth to onnx
onnx_model = onnx.load(f)
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))
print('ONNX export success, saved as %s' % f)