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

import torch.nn.parallel
import numpy as np
from ops.dataset import TSNDataSet
import torchvision
from ops.transforms import GroupScale
from ops.transforms import GroupCenterCrop
from ops.transforms import Stack
from ops.transforms import ToTorchFormatTensor
from ops.transforms import GroupNormalize
from ops import dataset_config
import mindx.sdk as sdk

WEIGHT = 'TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth'
weights = WEIGHT.split(',')
coeff = [1] * len(weights)
test_file = [None] * len(weights)
modalities = []
data_list = []
TOTAL = None


def process(video, segment, mol):
    with torch.no_grad():
        j, datas, labels = video
        batch_size = labels.numel()
        data1 = datas.view(-1, 3, datas.size(2), datas.size(3))
        data1 = data1.view(batch_size, segment, 3, data1.size(2), data1.size(3))
        filepath = "./model/TSM.om"
        device_id = 0                          
        m = sdk.model(filepath, device_id)
        t = sdk.Tensor(np.array(data1))
        t.to_device(0)
        rsts = m.infer(t)
        rsts[0].to_host()
        rsts = rsts[0]
        rsts = np.array(rsts)
        rsts = rsts.reshape(batch_size, 1, -1).mean(1)
        rsts = torch.Tensor(rsts)
        rsts = rsts.data.cpu().numpy().copy()
        rsts = rsts.reshape(batch_size, classes)
        return j, rsts, labels

for test in test_file:
    MODALITY = 'RGB'
    modalities.append(MODALITY)
    classes, train_list, val_list, path, prefix = dataset_config.return_dataset('kinetics', MODALITY)
    
    dataset = torch.utils.data.DataLoader(
            TSNDataSet(path, val_list, num_segments=8, new_length=1, modality='RGB',
                       image_tmpl='img_{:05d}.jpg', test_mode=True, remove_missing = 1,
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.Compose([GroupScale(256), GroupCenterCrop(224), ]),
                           Stack(roll=('resnet50' in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=('resnet50' not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                       ]), dense_sample=False, twice_sample=False), batch_size=1, shuffle=False, num_workers=8, 
                       pin_memory=True, )
    data_gen = enumerate(dataset)
    TOTAL = len(dataset.dataset)
    data_list.append(data_gen)
 

class Meter(object):
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


def acc(outputs, target, topk=(1,)):
    max1 = max(topk)
    batch = target.size(0)
    _, pred = outputs.topk(max1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct1 = correct[:k].view(-1).float().sum(0)
        res.append(correct1.mul_(100.0 / batch))
    return res

output = []
top1 = Meter()
top5 = Meter()

for i, data_label in enumerate(zip(*data_list)):
    with torch.no_grad():
        if i >= TOTAL:
            break
        this_rst = []
        THIS_LABEL = None
        for (_, (data, label)), modality in zip(data_label, modalities):
            rst = process((i, data, label), 8, modality)
            this_rst.append(rst[1])
            THIS_LABEL = label
        assert len(this_rst) == len(coeff)
        for i_coeff, this_rsts in enumerate(this_rst):
            this_rsts *= coeff[i_coeff]
        predict = sum(this_rst) / len(this_rst)

        for p, g in zip(predict, THIS_LABEL.cpu().numpy()):
            output.append([p[None, ...], g])
        prec1, prec5 = acc(torch.from_numpy(predict), THIS_LABEL, topk=(1, 5))
        top1.update(prec1.item(), THIS_LABEL.numel())
        top5.update(prec5.item(), THIS_LABEL.numel())
        print('video {} finished, finish {}/{}, Prec@1 {:.3f} Prec@5 {:.3f}'.format(i, i, TOTAL,
                                                                                    top1.avg, top5.avg))

print('-----finished------')
print('Finall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg))