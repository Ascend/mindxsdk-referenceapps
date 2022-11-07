# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

import torch
import torch.onnx
import torch.nn.parallel
import sys 
sys.path.append("..") 
from ops.models import TSN


def pth_to_onnx(input_shape, checkpoint, onnx_path):
    ckpt = torch.load(checkpoint)
    ckpt = ckpt['state_dict']
    base = {'.'.join(k.split('.')[1:]): v for k, v in list(ckpt.items())}
    replace = {'base_model.classifier.weight': 'new_fc.weight',
                    'base_model.classifier.bias': 'new_fc.bias',
                    }
    net = TSN(400, 8, 'RGB',
            base_model='resnet50',
            consensus_type='avg' ,
            img_feature_dim=256,
            pretrain='imagenet',
            is_shift=True, shift_div=8, shift_place='blockres',
            non_local=False,)
    for k, v in replace.items():
        if k in base:
            base[v] = base.pop(k)
    net.load_state_dict(base)
    net.eval()
    torch.onnx.export(net, input_shape, onnx_path, opset_version=11)
    print("Exporting .pth model to onnx model has been successful!")
    return None

if __name__ == '__main__':
    PTH = './TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth'
    ONNX_PATH = './TSM.onnx'
    INPUT_SHAPE = torch.randn(1, 8, 3, 224, 224)
    pth_to_onnx(INPUT_SHAPE, PTH, ONNX_PATH)

