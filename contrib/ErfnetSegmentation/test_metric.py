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
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi,  MxDataInput, StringVector, \
    InProtobufVector, MxProtobufIn
from tqdm import tqdm


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

    stream_name = b'erfnet'
    data_input.data = get_image_binary(img_path_)
    proto_buf_vec = InProtobufVector()
    vision_list = MxpiDataType.MxpiVisionList()
    vision_vec = vision_list.visionVec.add()
    vision_vec.visionInfo.format = 1
    vision_vec.visionData.deviceId = 0
    vision_vec.visionData.memType = 0
    vision_vec.visionData.dataStr = data_input.data
    protobuf = MxProtobufIn()
    protobuf.key = b'appsrc0'
    protobuf.type = b'MxTools.MxpiVisionList'
    protobuf.protobuf = vision_list.SerializeToString()
    proto_buf_vec.push_back(protobuf)
    unique_id = stream_manager_api_.SendProtobuf(stream_name, 0, proto_buf_vec)

    if unique_id < 0:
        print("Failed to send data to stream.")
        sys.exit()

    key = b'mxpi_tensorinfer0'
    key_vec = StringVector()
    key_vec.push_back(key)
    infer_result = stream_manager_api_.GetProtobuf(stream_name, 0, key_vec)
    if infer_result.size() == 0:
        print("infer_result is null")
        sys.exit()
    if infer_result[0].errorCode != 0:
        print('''GetResultWithUniqueId error. errorCode=%d''' % (
            infer_result[0].errorCode))
        sys.exit()
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    vision_data_ = result.tensorPackageVec[0].tensorVec[0].dataStr
    vision_data_ = np.frombuffer(vision_data_, dtype=np.float32)
    shape = result.tensorPackageVec[0].tensorVec[0].tensorShape
    vision_data_ = vision_data_.reshape(shape)
    return vision_data_


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

    def __getitem__(self, index):
        filename = self.filenames[index]
        filename_gt = self.filenames_gt[index]

        return filename, filename_gt

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--pipeline_path', type=str,
                        default="pipeline/erfnet_pipeline_for_metric.json")
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

    with open(pipeline_path, 'r') as file:
        json_str = file.read()

    pipeline = json.loads(json_str)

    pipelineStr = json.dumps(pipeline).encode()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        sys.exit()

    metrics = IouEval(num_classes=20)
    datasets = CityscapesValDatapath(data_path)
    for image_path, target_path in tqdm(datasets):
        with open(target_path, 'rb') as file:
            target = load_image(file).convert('P')
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

    with open("metric.txt", "w") as file:
        print("mean_iou: ", mean_iou, file=file)
        print("iou_class: ", iou_class, file=file)

    print("mean_iou: ", mean_iou)
    print("iou_class: ", iou_class)

    streamManagerApi.DestroyAllStreams()
