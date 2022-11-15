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

import argparse
import torch
from lib import VideoModel_pvtv2 as Network


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  type=str, default='MoCA')
parser.add_argument('--testsplit',  type=str, default='TestDataset_per_sq')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--trainsize', type=int, default=352, help='testing size')
parser.add_argument('--pretrained_cod10k', default=None, help='path to the pretrained Resnet')

parser.add_argument('--pth_path', type=str, default='./snapshot/Net_epoch_MoCA_short_term_pseudo.pth')
parser.add_argument('--onnx_path', type=str, default='./sltnet.onnx')

opt = parser.parse_args()


if __name__ == '__main__':
    model = Network(opt)

    model.load_state_dict(torch.load(opt.pth_path, map_location=torch.device('cpu')))
    model.eval()

    input_names = ["image"]  
    output_names = ["pred"]  
    dynamic_axes = {'image': {0: '-1'}, 'pred': {0: '-1'}} 
    dummy_input = torch.randn(1, 9, 352, 352)
    torch.onnx.export(model, dummy_input, opt.onnx_path, input_names=input_names, \
        dynamic_axes=dynamic_axes, output_names=output_names, opset_version=11, verbose=True) 
