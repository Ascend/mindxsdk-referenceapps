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
from io import BytesIO
import numpy as np
from PIL import Image
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi,  MxDataInput, StringVector, \
    InProtobufVector, MxProtobufIn
from tqdm import tqdm


def colormap_cityscapes():
    cmap = np.zeros([255 * 3]).astype(np.uint8)
    cmap[128 + 64 + 128] = 0
    cmap[244 + 35 + 232] = 1
    cmap[70 + 70 + 70] = 2
    cmap[102 + 102 + 156] = 3
    cmap[190 + 153 + 153] = 4
    cmap[153 + 153 + 153] = 5
    cmap[250 + 170 + 30] = 6
    cmap[220 + 220 + 0] = 7
    cmap[107 + 142 + 35] = 8
    cmap[152 + 251 + 152] = 9
    cmap[70 + 130 + 180] = 10
    cmap[220 + 20 + 60] = 11
    cmap[255 + 0 + 0] = 12
    cmap[0 + 0 + 142] = 13
    cmap[0 + 0 + 70] = 14
    cmap[0 + 60 + 100] = 15
    cmap[0 + 80 + 100] = 16
    cmap[0 + 0 + 230] = 17
    cmap[119 + 11 + 32] = 18
    cmap[0 + 0 + 0] = 19
    return cmap


def load_image(file_name):
    return Image.open(file_name)


def get_image_binary(img_path_):
    with open(img_path_, 'rb') as file__:
        image = Image.open(file__).convert('RGB')
    image = resize(image, 512, Image.BILINEAR)
    image = np.array(image).astype(np.float32) / 255
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    return image.tobytes()


def infer(img_path_, stream_manager_api_):
    data_input = MxDataInput()
    with open(img_path_, 'rb') as file__:
        image = Image.open(file__)
        output = BytesIO()
        image.save(output, format='JPEG')
        data_input.data = output.getvalue()
    unique_id = stream_manager_api_.SendData(b'erfnet', 0, data_input)
    if unique_id < 0:
        print("Failed to send data to stream.")


def resize(img, size, interpolation):
    if isinstance(size, int):
        width, height = img.size
        if (width <= height and width == size) or (height <= width and height == size):
            return img
        if width < height:
            o_width = size
            o_height = int(size * height / width)
            return img.resize((o_width, o_height), interpolation)
        o_height = size
        o_width = int(size * width / height)
        return img.resize((o_width, o_height), interpolation)
    return img.resize(size[::-1], interpolation)


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

    def __getitem__(self, index_):
        filename = self.filenames[index_]
        filename_gt = self.filenames_gt[index_]

        return filename, filename_gt

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--pipeline_path', type=str,
                        default="pipeline/erfnet_pipeline.json")
    parser.add_argument('--data_path', type=str)
    config = parser.parse_args()

    pipeline_path = config.pipeline_path
    data_path = config.data_path

    INFER_RESULT = "infer_result/"
    if not os.path.exists(INFER_RESULT):
        os.mkdir(INFER_RESULT)

    datapath = CityscapesValDatapath(data_path)

    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        sys.exit()

    with open(pipeline_path, 'r') as file:
        json_str = file.read()

    pipeline = json.loads(json_str)

    pipelineStr = json.dumps(pipeline).encode()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        sys.exit()

    color2class = colormap_cityscapes()

    metrics = IouEval(num_classes=20)
    datasets = CityscapesValDatapath(data_path)
    print(len(datasets))
    for index, (image_path, target_path) in tqdm(enumerate(datasets)):
        print(index, image_path)
        print(index, target_path)
        infer(image_path, streamManagerApi)
        with open(target_path, 'rb') as file:
            target = load_image(file).convert('P')
        target = resize(target, 512, Image.NEAREST)
        target = np.array(target).astype(np.uint32)
        target[target == 255] = 19
        target = target.reshape(512, 1024)
        target = target[np.newaxis, :, :]

        RESIMAGE = INFER_RESULT + str(index) + ".png"
        while True:  # 轮询, 等待异步线程
            try:
                preds = Image.open(RESIMAGE).convert('RGB')
                break
            except:
                continue
            continue
        preds = np.array(preds)
        preds = preds.transpose(2, 0, 1)
        preds = np.expand_dims(preds, 0).astype(np.uint8)
        preds = preds.sum(axis=1)
        preds = np.expand_dims(preds, 0)
        preds = color2class[preds]
        labels_ = np.expand_dims(target.astype(np.int32), 1).astype(np.int32)
        metrics.add_batch(preds, labels_)

    mean_iou, iou_class = metrics.get_iou()
    mean_iou = mean_iou.item()
    print("mean_iou: ", mean_iou)
    print("iou_class: ", iou_class)

    streamManagerApi.DestroyAllStreams()
