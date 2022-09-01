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
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector, \
    InProtobufVector, MxProtobufIn
from tqdm import tqdm
import imageio


def my_open(file_path, asdads):
    fd_asd = os.open(file_path, os.O_RDWR | os.O_CREAT)
    file__22 = os.fdopen(fd_asd, asdads)
    return file__22


def resize(img, size, interpolation=2, max_size=None):
    if isinstance(size, int):
        wing, s_h_w = img.size

        short, long = (wing, s_h_w) if wing <= s_h_w else (s_h_w, wing)
        new_short, new_long = size, int(size * long / short)

        if max_size is not None:
            if new_long > max_size:
                new_short, new_long = int(
                    max_size * new_short / new_long), max_size

        new_w, new_h = (new_short, new_long) if wing <= s_h_w else (
            new_long, new_short)

        if (wing, s_h_w) == (new_w, new_h):
            return img
        return img.resize((new_w, new_h), interpolation)
    return img.resize(size[::-1], interpolation)


def infer(data_, stream_manager_api):
    data_input = MxDataInput()

    stream_name = b'pranet'
    data_input.data = data_
    protobuf_vec = InProtobufVector()
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
    protobuf_vec.push_back(protobuf)
    unique_id = stream_manager_api.SendProtobuf(stream_name, 0, protobuf_vec)

    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()

    key = b'mxpi_tensorinfer0'
    key_vec = StringVector()
    key_vec.push_back(key)
    infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
    if infer_result.size() == 0:
        print("infer_result is null")
        exit()
    if infer_result[0].errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d" % (
            infer_result[0].errorCode))
        exit()

    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    vision_data_ = result.tensorPackageVec[0].tensorVec[3].dataStr
    vision_data_ = np.frombuffer(vision_data_, dtype=np.float32)
    shape_ = result.tensorPackageVec[0].tensorVec[3].tensorShape
    vision_data_ = vision_data_.reshape(shape_)
    return vision_data_


def rgb_loader(path):

    with my_open(path, 'rb') as file_:
        img = Image.open(file_)
        return img.convert('RGB')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--pipeline_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type=str)
    config = parser.parse_args()

    pipeline_path = config.pipeline_path
    data_path = config.data_path
    output_path = config.output_path

    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        sys.exit()

    with my_open(pipeline_path, "r") as file:
        json_str = file.read()
    pipeline = json.loads(json_str)
    pipelineStr = json.dumps(pipeline).encode()

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        sys.exit()

    mean = np.array([[[0.485]], [[0.456]], [[0.406]]], dtype=np.float32)
    std = np.array([[[0.229]], [[0.224]], [[0.225]]], dtype=np.float32)

    if len(os.listdir(data_path)) == 0:
        raise RuntimeError("No Input Image!")

    for data in tqdm(os.listdir(data_path)):
        image_path = os.path.join(data_path, data)
        image = rgb_loader(image_path)
        shape = image.size

        image = resize(image, (352, 352))  # resize
        image = np.transpose(image, (2, 0, 1)).astype(
            np.float32)  # to tensor 1
        image = image / 255  # to tensor 2
        image = (image - mean) / std  # normalize
        res = infer(image.tobytes(), streamManagerApi)

        res = res.reshape((352, 352))
        res = res.T
        res = np.expand_dims(res, 0)
        res = np.expand_dims(res, 0)
        res = 1 / (1 + np.exp(-res))
        res = res.squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        _, name = os.path.split(image_path)
        imageio.imwrite(os.path.join(output_path, name), res)

    streamManagerApi.DestroyAllStreams()
