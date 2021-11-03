# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True
    
def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

# 载入所需进行转换的PyTorch模型
cfg = cfg_mnet
model = Retina(cfg=cfg, phase='test')
model = load_model(model, opt.weights , False)
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

