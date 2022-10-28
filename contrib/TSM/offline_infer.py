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
import os
import time
import torch.nn.parallel
import torch.optim
import numpy as np
from ops.dataset import TSNDataSet
import torchvision
from ops.transforms import GroupScale
from ops.transforms import GroupCenterCrop
from ops.transforms import Stack
from ops.transforms import ToTorchFormatTensor
from ops.transforms import GroupNormalize
from ops import dataset_config
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
import mindx.sdk as sdk

weights = 'TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth'
weights_list = weights.split(',')
coeff_list = [1] * len(weights_list)
test_file_list = [None] * len(weights_list)
modality_list = []
data_iter_list = []
TOTAL_NUM = None
for this_weights, test_file in zip(weights_list, test_file_list):
    MODALITY = 'RGB'
    modality_list.append(MODALITY)
    num_class, train_list, val_list, root_path, prefix = dataset_config.return_dataset('kinetics', MODALITY)
    cropping = torchvision.transforms.Compose([
            GroupScale(256),
            GroupCenterCrop(224),
        ])    
    
    data_loader = torch.utils.data.DataLoader(
            TSNDataSet(root_path, test_file if test_file is not None else val_list, num_segments=8,
                       new_length=1 if MODALITY == "RGB" else 5,
                       modality='RGB',
                       image_tmpl='img_{:05d}.jpg',
                       test_mode=True,
                       remove_missing = len(weights_list) == 1,
                       transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=('resnet50' in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=('resnet50' not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                       ]), dense_sample=False, twice_sample=False),
            batch_size=1, shuffle=False,
            num_workers=8, pin_memory=True,
    )

    data_gen = enumerate(data_loader)

    if TOTAL_NUM is None:
        TOTAL_NUM = len(data_loader.dataset)
    else:
        assert TOTAL_NUM == len(data_loader.dataset)

    data_iter_list.append(data_gen)
 

class AverageMeter(object):
    def __init__(self):
        self.reset()
        self.val = None
        self.avg = None

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def eval_video(video_data, test_segments, mol):
    with torch.no_grad():
        j, datas, labels = video_data
        batch_size = labels.numel()
        num_crop = 1
        length = 3

        data_in = datas.view(-1, length, datas.size(2), datas.size(3))
        data_in = data_in.view(batch_size * num_crop, test_segments, length, data_in.size(2), data_in.size(3))
        filepath = "../model/TSM.om"
        device_id = 0                          
        m = sdk.model(filepath, device_id)
        t = sdk.Tensor(np.array(data_in))
        t.to_device(0)
        rsts = m.infer(t)
        rsts[0].to_host()
        rsts = rsts[0]
        rsts = np.array(rsts)
        rsts = rsts.reshape(batch_size, num_crop, -1).mean(1)
        rsts = torch.Tensor(rsts)
        rsts = rsts.data.cpu().numpy().copy()

        
        rsts = rsts.reshape(batch_size, num_class)

        return j, rsts, labels

def accuracy(outputs, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

output = []
proc_start_time = time.time()
max_num = TOTAL_NUM

top1 = AverageMeter()
top5 = AverageMeter()


for i, data_label_pairs in enumerate(zip(*data_iter_list)):
    with torch.no_grad():
        if i >= max_num:
            break
        this_rst_list = []
        THIS_LABEL = None
        for (_, (data, label)), modality in zip(data_label_pairs, modality_list):
            rst = eval_video((i, data, label), 8, modality)
            this_rst_list.append(rst[1])
            THIS_LABEL = label
        assert len(this_rst_list) == len(coeff_list)
        for i_coeff, this_rst in enumerate(this_rst_list):
            this_rst *= coeff_list[i_coeff]
        ensembled_predict = sum(this_rst_list) / len(this_rst_list)

        for p, g in zip(ensembled_predict, THIS_LABEL.cpu().numpy()):
            output.append([p[None, ...], g])
        cnt_time = time.time() - proc_start_time
        prec1, prec5 = accuracy(torch.from_numpy(ensembled_predict), THIS_LABEL, topk=(1, 5))
        top1.update(prec1.item(), THIS_LABEL.numel())
        top5.update(prec5.item(), THIS_LABEL.numel())
        print('video {} done, total {}/{}, average {:.3f} sec/video, '
              'moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i, i, TOTAL_NUM,
                                                        float(cnt_time) / (i+1), top1.avg, top5.avg))

print('-----Evaluation is finished------')
print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg))