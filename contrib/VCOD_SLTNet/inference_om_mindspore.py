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
import numpy as np
from tqdm import tqdm
import imageio
from PIL import Image
from mindx.sdk.base import Tensor, Model
import mindspore.dataset.vision as vision
import imageio


parser = argparse.ArgumentParser()
parser.add_argument('--testsplit',  type=str, default='TestDataset_per_sq')
parser.add_argument('--datapath',  type=str, default='./SLT-Net/')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--trainsize', type=int, default=352, help='testing size')
parser.add_argument('--pretrained_cod10k', default=None, help='path to the pretrained Resnet')
 
parser.add_argument('--pth_path', type=str, default='./Net_epoch_MoCA_short_term_pseudo.pth')
parser.add_argument('--onnx_path', type=str, default='./sltnet.onnx')

parser.add_argument('--save_root', type=str, default='./sltnet_res/')
parser.add_argument('--om_path', type=str, default='./om_model/sltnet.om')
parser.add_argument('--device_id', type=int, default=0)

opt = parser.parse_args()


class test_dataset:
    def __init__(self, split='TestDataset_per_sq', datapath='dataset', testsize=352):
        self.testsize = testsize
        self.image_list = []
        self.gt_list = []
        self.extra_info = []

        root = datapath
        img_format = '*.jpg'
        data_root = os.path.join(root, split)
        print(split)

        for scene in os.listdir(os.path.join(data_root)):
            if split =='MoCA-Video-Test':
                images  = sorted(glob(os.path.join(data_root, scene, 'Frame', img_format)))
            elif split =='TestDataset_per_sq':
                images = sorted(glob(os.path.join(data_root, scene, 'Imgs', img_format)))
            gt_list = sorted(glob(os.path.join(data_root, scene, 'GT', '*.png')))

            for j in range(len(images)-2):
                self.extra_info += [ (scene, j) ]  # scene and frame_id
                self.gt_list    += [ gt_list[j] ]
                self.image_list += [ [images[j],
                                    images[j+1],
                                    images[j+2]] ]

        self.resize = vision.Resize(size=(self.testsize, self.testsize))
        self.norm = vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.index = 0
        self.size = len(self.gt_list)

    def load_data(self):
        imgs = []
        names_img = []

        for i in range(len(self.image_list[self.index])):
            imgs += [self.rgb_loader(self.image_list[self.index][i])]
            names_img += [self.image_list[self.index][i].split('/')[-1]]
            imgs[i] = self.resize(imgs[i])
            imgs[i] = self.norm(imgs[i])
            imgs[i] = np.ascontiguousarray(imgs[i], dtype=np.float32)

        scene_idx = self.image_list[self.index][0].split('/')[-3]  
        gt_idx = self.binary_loader(self.gt_list[self.index])

        self.index += 1
        self.index = self.index % self.size
    
        return imgs, gt_idx, names_img, scene_idx

    @staticmethod
    def rgb_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    @staticmethod
    def binary_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


if __name__ == '__main__':
    test_loader = test_dataset(split=opt.testsplit, datapath=opt.datapath, testsize=opt.testsize)

    model = Model(opt.om_path, opt.device_id)

    for i in tqdm(range(test_loader.size)):
        images_in, gt, names, scene_name = test_loader.load_data()
        save_path = opt.save_root + scene_name + '/Pred/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        model_in = np.concatenate(images_in, axis=-1)[..., None].transpose((3, 2, 0, 1))
        model_in = np.ascontiguousarray(model_in, dtype=np.float32)
        model_in = Tensor(model_in)

        model_in.to_device(opt.device_id)
        out = model.infer(model_in)
        out = out[0]
        out.to_host()
        res = np.array(out).squeeze()
        res = vision.Resize(size=gt.shape)(res)
        res = 1/(1+(np.exp((-res))))
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)

        name = names[0].replace('jpg', 'png')

        fp = save_path + name
        imageio.imwrite(fp, res)

        print('> ', fp)
