import argparse
import os
import sys
import torch
#以下两条语句为导入模型的网络结构，因为本项目的pth文件只保存了模型的权重参数
from models.retina import Retina 
from data import cfg_mnet

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='./weights/mnet_plate.pth', help='weights path')
parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
opt = parser.parse_args()

# 载入所需进行转换的PyTorch模型
cfg = cfg_mnet
model = Retina(cfg=cfg, phase='test')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_dict = torch.load(opt.weights, map_location=lambda storage, loc: storage.cuda(device))
model.load_state_dict(pretrained_dict, strict=False)
model.eval()

# 构建模型的输入
img = torch.randn((opt.batch_size, 3, *opt.img_size))

# 转换后的onnx模型的文件名
f = opt.weights.replace('.pth', '.onnx')  # filename

# ONNX export
try:
    import onnx

    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)    
    torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['image']) # pth to onnx

    # 对转换所得的onnx模型进行校验
    onnx_model = onnx.load(f)  # 载入onnx模型
    onnx.checker.check_model(onnx_model)  # 校验onnx模型
    print(onnx.helper.printable_graph(onnx_model.graph))  # 打印onnx模型的结构
    print('ONNX export success, saved as %s' % f)
except Exception as e:
    print('ONNX export failure: %s' % e)