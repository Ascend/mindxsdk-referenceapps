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
import json
import os
from argparse import ArgumentParser
import numpy as np
from PIL import Image
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector, InProtobufVector, MxProtobufIn
from tqdm import tqdm

from infer import resize, getImgB, colormap_cityscapes, load_image, infer


def is_image(filename):
    return any(filename.endswith(ext) for ext in ['.jpg', '.png'])


def is_label(filename):
    return filename.endswith("_labelTrainIds.png")


class IouEval:

    def __init__(self, NumClasses, ignoreIndex=19):
        self.NumClasses = NumClasses
        self.ignoreIndex = ignoreIndex if NumClasses > ignoreIndex else -1
        self.reset()

    def reset(self):
        classes = self.NumClasses if self.ignoreIndex == -1 else self.NumClasses-1
        self.tp = np.zeros([classes], dtype=np.float32)
        self.fp = np.zeros([classes], dtype=np.float32)
        self.fn = np.zeros([classes], dtype=np.float32)

    def addBatch(self, x_, y_):

        def toOneHot(tensor):
            if tensor.shape[1] == 1:
                pixnum = tensor.shape[2] * tensor.shape[3]
                onehot = np.zeros((self.NumClasses, pixnum))
                onehot[tensor.reshape(-1), np.arange(pixnum)] = 1
                onehot = onehot.reshape(
                    (1, self.NumClasses, tensor.shape[2], tensor.shape[3]))
                onehot = onehot.astype(np.float64)
            else:
                onehot = tensor.float()
            return onehot

        x_onehot_ = toOneHot(x_)
        y_onehot_ = toOneHot(y_)

        if self.ignoreIndex != -1:
            ignores_ = np.expand_dims(y_onehot_[:, self.ignoreIndex], 1)
            x_onehot_ = x_onehot_[:, :self.ignoreIndex]
            y_onehot_ = y_onehot_[:, :self.ignoreIndex]
        else:
            ignores_ = 0

        def agragate(tensor):
            res_ = np.sum(
                np.sum(np.sum(tensor, axis=0, keepdims=True),
                       axis=2, keepdims=True),
                axis=3, keepdims=True
            )
            return np.squeeze(res_)

        tpmult_ = x_onehot_ * y_onehot_
        tp_ = agragate(tpmult_)

        fpmult_ = x_onehot_ * (1-y_onehot_-ignores_)
        fp_ = agragate(fpmult_)

        fnmult_ = (1-x_onehot_) * (y_onehot_)
        fn_ = agragate(fnmult_)

        self.tp += tp_.astype(np.float64)
        self.fp += fp_.astype(np.float64)
        self.fn += fn_.astype(np.float64)

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return np.mean(iou), iou


class cityscapes_val_datapath:

    def __init__(self, root):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')

        subset = "val"
        self.images_root += subset
        self.labels_root += subset

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in
                          os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in
                            os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        return filename, filenameGt

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--pipeline_path', type=str)
    parser.add_argument('--data_path', type=str)
    config = parser.parse_args()

    pipeline_path = config.pipeline_path
    data_path = config.data_path

    datapath = cityscapes_val_datapath(data_path)

    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    with os.open(pipeline_path, "r") as file:
        json_str = file.read()
    pipeline = json.loads(json_str)

    pipelineStr = json.dumps(pipeline).encode()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    metrics = iouEval_1(NumClasses=20)
    datasets = cityscapes_val_datapath(data_path)
    for image_path, target_path in tqdm(datasets):
        with os.open(target_path, 'rb') as f:
            target = load_image(f).convert('P')
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
        metrics.addBatch(preds_, labels_)

    mean_iou, iou_class = metrics.getIoU()
    mean_iou = mean_iou.item()
    with os.open("metric.txt", "w") as file:
        print("mean_iou: ", mean_iou, file=file)
        print("iou_class: ", iou_class, file=file)
    print("mean_iou: ", mean_iou)
    print("iou_class: ", iou_class)

    streamManagerApi.DestroyAllStreams()
