import os
import cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special 
import tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
from PIL import Image

# Export to TorchScript that can be used for LibTorch

torch.backends.cudnn.benchmark = True

# From cuLANE, Change this line if you are using TuSimple
cls_num_per_lane = 18
griding_num = 200
backbone = 18

net = parsingNet(pretrained = False,backbone='18', cls_dim = (griding_num+1,cls_num_per_lane,4),use_aux=False)

# Change test_model where your model stored.
test_model = r'./model/culane_18.pth'

state_dict = torch.load(test_model, map_location='cpu')['model'] # CPU
compatible_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        compatible_state_dict[k[7:]] = v
    else:
        compatible_state_dict[k] = v

net.load_state_dict(compatible_state_dict, strict=False)
net.eval()

# Test Input Image
img = torch.zeros(1, 3, 288, 800)  # image size(1,3,320,192) iDetection
y = net(img)  # dry run

try:
    import onnx

    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = './model/culane_18.pth'.replace('.pth', '.onnx')  # filename
    torch.onnx.export(net, img, f, verbose=False, opset_version=11, input_names=['images'],output_names=['classes', 'boxes'] if y is None else ['output'])

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    print('ONNX export success, saved as %s' % f)
except Exception as e:
    print('ONNX export failure: %s' % e)
