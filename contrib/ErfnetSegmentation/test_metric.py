# Copyright (c) 2022. Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import json
from argparse import ArgumentParser
import numpy as np
from PIL import Image
from StreamManagerApi import StreamManagerApi
from tqdm import tqdm

from infer import resize, load_image, infer


def is_image(filename):
    return any(filename.endswith(ext) for ext in ['.jpg', '.png'])


def is_label(filename):
    return filename.endswith("_labelTrainIds.png")


class IouEval:

    def __init__(self, num_classes, ignore_index=19):
        self.num_classes = num_classes
        self.ignore_index = ignore_index if num_classes > ignore_index else -1
        self.reset()

    def reset(self):
        classes = self.num_classes if self.ignore_index == -1 else self.num_classes-1
        self.t_p = np.zeros([classes], dtype=np.float32)
        self.f_p = np.zeros([classes], dtype=np.float32)
        self.f_n = np.zeros([classes], dtype=np.float32)

    def add_batch(self, xing, ying):

        def to_one_hot(tensor):
            if tensor.shape[1] == 1:
                pixnum = tensor.shape[2] * tensor.shape[3]
                onehot = np.zeros((self.num_classes, pixnum))
                onehot[tensor.reshape(-1), np.arange(pixnum)] = 1
                onehot = onehot.reshape(
                    (1, self.num_classes, tensor.shape[2], tensor.shape[3]))
                onehot = onehot.astype(np.float64)
            else:
                onehot = tensor.float()
            return onehot

        xonehot_ = to_one_hot(xing)
        yonehot_ = to_one_hot(ying)

        if self.ignore_index != -1:
            ignores_ = np.expand_dims(yonehot_[:, self.ignore_index], 1)
            xonehot_ = xonehot_[:, :self.ignore_index]
            yonehot_ = yonehot_[:, :self.ignore_index]
        else:
            ignores_ = 0

        def agragate(tensor):
            res_ = np.sum(
                np.sum(np.sum(tensor, axis=0, keepdims=True),
                       axis=2, keepdims=True),
                axis=3, keepdims=True
            )
            return np.squeeze(res_)

        tpmult_ = xonehot_ * yonehot_
        tp_ = agragate(tpmult_)

        fpmult_ = xonehot_ * (1-yonehot_-ignores_)
        fp_ = agragate(fpmult_)

        fnmult_ = (1-xonehot_) * (yonehot_)
        fn_ = agragate(fnmult_)

        self.t_p += tp_.astype(np.float64)
        self.f_p += fp_.astype(np.float64)
        self.f_n += fn_.astype(np.float64)

    def get_iou(self):
        num = self.t_p
        den = self.t_p + self.f_p + self.f_n + 1e-15
        iou = num / den
        return np.mean(iou), iou


class CityscapesValDatapath:

    def __init__(self, root):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')

        subset = "val"
        self.images_root += subset
        self.labels_root += subset

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in
                          os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        self.filenames_gt = [os.path.join(dp, f) for dp, dn, fn in
                             os.walk(os.path.expanduser(self.labels_root))
                             for f in fn if is_label(f)]
        self.filenames_gt.sort()

    def __getitem__(self, index):
        filename = self.filenames[index]
        filename_gt = self.filenames_gt[index]

        return filename, filename_gt

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--pipeline_path', type=str)
    parser.add_argument('--data_path', type=str)
    config = parser.parse_args()

    pipeline_path = config.pipeline_path
    data_path = config.data_path

    datapath = CityscapesValDatapath(data_path)

    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        sys.exit()

    fd = os.open(pipeline_path, os.O_RDWR | os.O_CREAT)
    file = os.fdopen(fd, "r")
    json_str = file.read()
    file.close()

    pipeline = json.loads(json_str)

    pipelineStr = json.dumps(pipeline).encode()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        sys.exit()

    metrics = IouEval(num_classes=20)
    datasets = CityscapesValDatapath(data_path)
    for image_path, target_path in tqdm(datasets):
        fd = os.open(target_path, os.O_RDWR | os.O_CREAT)
        file = os.fdopen(fd, 'rb')
        target = load_image(file).convert('P')
        file.close()

        target = resize(target, 512, Image.NEAREST)
        target = np.array(target).astype(np.uint32)
        target[target == 255] = 19
        target = target.reshape(512, 1024)
        target = target[np.newaxis, :, :]

        res = infer(image_path, streamManagerApi)
        res = res.reshape(1, 20, 512, 1024)

        preds_ = np.expand_dims(res.argmax(
            axis=1).astype(np.int32), 1).astype(np.int32)
        labels_ = np.expand_dims(target.astype(np.int32), 1).astype(np.int32)
        metrics.add_batch(preds_, labels_)

    mean_iou, iou_class = metrics.get_iou()
    mean_iou = mean_iou.item()

    fd = os.open("metric.txt", os.O_RDWR | os.O_CREAT)
    file = os.fdopen(fd, "w")
    print("mean_iou: ", mean_iou, file=file)
    print("iou_class: ", iou_class, file=file)
    file.close()

    print("mean_iou: ", mean_iou)
    print("iou_class: ", iou_class)

    streamManagerApi.DestroyAllStreams()
