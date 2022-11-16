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


class test_dataset:
    def __init__(self, datapath, testsize):
        self.testsize = testsize
        self.image_list = []
        self.gt_list = []
        self.extra_info = []

        img_format = '*.jpg'
        data_root = os.path.join(datapath, 'TestDataset_per_sq')

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        for scene in os.listdir(os.path.join(data_root)):
            images  = sorted(glob(os.path.join(data_root, scene, 'Imgs', img_format)))
            gt_list = sorted(glob(os.path.join(data_root, scene, 'GT', '*.png')))

            for i in range(len(images)-2):
                self.extra_info += [ (scene, i) ]
                self.gt_list    += [ gt_list[i] ]
                self.image_list += [ [images[i], 
                                    images[i+1], 
                                    images[i+2]] ]

        self.index = 0
        self.size = len(self.gt_list)

    def load_data(self):
        imgs = []
        names= []

        for i in range(len(self.image_list[self.index])):
            imgs += [self.rgb_loader(self.image_list[self.index][i])]
            names+= [self.image_list[self.index][i].split('/')[-1]]

            imgs[i] = cv2.resize(imgs[i], (self.testsize, self.testsize))
            imgs[i] = np.array([imgs[i]])
            imgs[i] = imgs[i].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            imgs[i] = (imgs[i] - np.asarray(self.mean)[None, :, None, None]) / np.asarray(self.std)[None, :, None, None]

        scene= self.image_list[self.index][0].split('/')[-3]  
        gt = self.binary_loader(self.gt_list[self.index])

        self.index += 1
        self.index = self.index % self.size
    
        return imgs, gt, names, scene

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

    def __len__(self):
        return self.size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath',  type=str, default='./data/')
    parser.add_argument('--save_root', type=str, default='./results/')
    parser.add_argument('--om_path', type=str, default='sltnet.om')
    parser.add_argument('--testsize', type=int, default=352)
    parser.add_argument('--device_id', type=int, default=0)
    opt = parser.parse_args()

    test_loader = test_dataset(datapath=opt.datapath, testsize=opt.testsize)

    model = Model(opt.om_path, opt.device_id)

    for i in tqdm(range(test_loader.size)):
        images, gt, names, scene = test_loader.load_data()
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

        name = names[0].replace('jpg', 'png')
        fp = save_path + name
        imageio.imwrite(fp, res)
        print('> ', fp)
