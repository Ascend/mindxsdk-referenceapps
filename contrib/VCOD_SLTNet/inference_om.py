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

import os
import argparse
import numpy as np
from tqdm import tqdm
import imageio
import torch
import torch.nn.functional as F
from dataloaders import test_dataloader
from mindx.sdk.base import Tensor, Model
import mindspore.ops as ops


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  type=str, default='MoCA')
parser.add_argument('--testsplit',  type=str, default='TestDataset_per_sq')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--trainsize', type=int, default=352, help='testing size')
parser.add_argument('--pretrained_cod10k', default=None, help='path to the pretrained Resnet')

parser.add_argument('--pth_path', type=str, default='./Net_epoch_MoCA_short_term_pseudo.pth')
parser.add_argument('--onnx_path', type=str, default='./sltnet.onnx')

parser.add_argument('--save_root', type=str, default='./sltnet_res/')
parser.add_argument('--om_path', type=str, default='./om_model/sltnet.om')
parser.add_argument('--device_id', type=int, default=0)

opt = parser.parse_args()


if __name__ == '__main__':
    test_loader = test_dataloader(opt)

    model = Model(opt.om_path, opt.device_id)

    for i in tqdm(range(test_loader.size)):
        images, gt, names, scene = test_loader.load_data()
        save_path = opt.save_root + scene + '/Pred/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        images = torch.cat(images, dim=1)

        images = images.numpy()
        imageTensor = Tensor(images)
        imageTensor.to_device(opt.device_id)
        out = model.infer(imageTensor)
        res = out[-1]
        res.to_host()

        res = torch.from_numpy(np.array(res))

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        name = names[0].replace('jpg', 'png')

        fp = save_path + name
        imageio.imwrite(fp, res)

        print('> ', fp)
