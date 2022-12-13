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
import sys
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import imageio
import cv2
from PIL import Image
import imageio
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindx.sdk.base import Tensor, Model


class TestDataset:
    def __init__(self, datapath, testsize):
        self.testsize = testsize
        self.image_list = []
        self.gt_list = []
        self.extra_info = []

        img_format = '*.jpg'
        data_root = os.path.join(datapath, 'TestDataset_per_sq')

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        for scene_i in os.listdir(os.path.join(data_root)):
            images_i  = sorted(glob(os.path.join(data_root, scene_i, 'Imgs', img_format)))
            gt_list = sorted(glob(os.path.join(data_root, scene_i, 'GT', '*.png')))

            for ii in range(len(images_i)-2):
                self.extra_info += [ (scene_i, ii) ]
                self.gt_list    += [ gt_list[ii] ]
                self.image_list += [ [images_i[ii], 
                                    images_i[ii+1], 
                                    images_i[ii+2]] ]

        self.index = 0
        self.size = len(self.gt_list)

    def __len__(self):
        return self.size
        
    @staticmethod
    def rgb_loader(path):
        image_bgr = cv2.imread(path)
        imge_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return imge_rgb

    @staticmethod
    def binary_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def load_data(self):
        imgs = []
        names = []

        for idx in range(len(self.image_list[self.index])):
            imgs += [self.rgb_loader(self.image_list[self.index][idx])]
            names += [self.image_list[self.index][idx].split('/')[-1]]

            imgs[idx] = cv2.resize(imgs[idx], (self.testsize, self.testsize))
            imgs[idx] = np.array([imgs[idx]])
            imgs[idx] = imgs[idx].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            imgs[idx] = (imgs[idx] - np.asarray(self.mean)[None, :, None, None]) / \
                np.asarray(self.std)[None, :, None, None]

        scenes = self.image_list[self.index][0].split('/')[-3]  
        gt_i = self.binary_loader(self.gt_list[self.index])

        self.index += 1
        self.index = self.index % self.size
    
        return {'imgs': imgs, 'gt': gt_i, 'names': names, 'scenes': scenes}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath',  type=str, default='./data/')
    parser.add_argument('--save_root', type=str, default='./results/')
    parser.add_argument('--om_path', type=str, default='sltnet.om')
    parser.add_argument('--testsize', type=int, default=352)
    parser.add_argument('--device_id', type=int, default=0)
    opt = parser.parse_args()

    if not opt.om_path.endswith('om'):
        print("Please check the correctness of om file:", opt.om_path)
        sys.exit()

    if 'TestDataset_per_sq' not in os.listdir(opt.datapath):
        print("Please check the correctness of dataset path:", opt.datapath)
        sys.exit()

    test_loader = TestDataset(datapath=opt.datapath, testsize=opt.testsize)

    model = Model(opt.om_path, opt.device_id)

    for i in tqdm(range(test_loader.size)):
        dataset = test_loader.load_data()
        images, gt, name, scene = dataset.get('imgs'), dataset.get('gt'), \
            dataset.get('names'), dataset.get('scenes')
        gt = np.asarray(gt, np.float32)
        save_path = opt.save_root + scene + '/Pred/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_in = np.concatenate(images, axis=1)
        model_in = np.ascontiguousarray(model_in, dtype=np.float32)
        model_in = Tensor(model_in)
        model_in.to_device(opt.device_id)
        out = model.infer(model_in)
        out = out[0]
        out.to_host()
        res = np.array(out)

        res = ms.Tensor(res)
        res = ops.Sigmoid()(res)
        res = nn.ResizeBilinear()(res, (gt.shape[0], gt.shape[1]))
        res = (res - res.min()) / (res.max() - res.min() + 1e-8) * 255
        res = res.astype('uint8')
        res = res.asnumpy().squeeze()

        name = name[0].replace('jpg', 'png')
        fp = save_path + name
        imageio.imwrite(fp, res)
        print('> ', fp)
