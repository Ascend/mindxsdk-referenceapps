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
import cv2
from tqdm import tqdm


def resize(img, size, interpolation):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh), interpolation)
    return img.resize(size[::-1], interpolation)


def getImgB(img_path_):
    with open(img_path_, 'rb') as f_:
        image = Image.open(f_).convert('RGB')
    image = resize(image, 512, Image.BILINEAR)
    image = np.array(image).astype(np.float32) / 255
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    return image.tobytes()


def colormap_cityscapes():
    cmap = np.zeros([20, 3]).astype(np.uint8)
    cmap[0, :] = np.array([128, 64, 128])
    cmap[1, :] = np.array([244, 35, 232])
    cmap[2, :] = np.array([70, 70, 70])
    cmap[3, :] = np.array([102, 102, 156])
    cmap[4, :] = np.array([190, 153, 153])
    cmap[5, :] = np.array([153, 153, 153])

    cmap[6, :] = np.array([250, 170, 30])
    cmap[7, :] = np.array([220, 220, 0])
    cmap[8, :] = np.array([107, 142, 35])
    cmap[9, :] = np.array([152, 251, 152])
    cmap[10, :] = np.array([70, 130, 180])

    cmap[11, :] = np.array([220, 20, 60])
    cmap[12, :] = np.array([255, 0, 0])
    cmap[13, :] = np.array([0, 0, 142])
    cmap[14, :] = np.array([0, 0, 70])
    cmap[15, :] = np.array([0, 60, 100])

    cmap[16, :] = np.array([0, 80, 100])
    cmap[17, :] = np.array([0, 0, 230])
    cmap[18, :] = np.array([119, 11, 32])
    cmap[19, :] = np.array([0, 0, 0])
    return cmap


def load_image(file_name):
    return Image.open(file_name)


def infer(img_path_, stream_manager_api_):
    dataInput = MxDataInput()

    streamName = b'erfnet'
    dataInput.data = getImgB(img_path_)
    protobufVec = InProtobufVector()
    visionList = MxpiDataType.MxpiVisionList()
    visionVec = visionList.visionVec.add()
    visionVec.visionInfo.format = 1
    visionVec.visionData.deviceId = 0
    visionVec.visionData.memType = 0
    visionVec.visionData.dataStr = dataInput.data
    protobuf = MxProtobufIn()
    protobuf.key = b'appsrc0'
    protobuf.type = b'MxTools.MxpiVisionList'
    protobuf.protobuf = visionList.SerializeToString()
    protobufVec.push_back(protobuf)
    uniqueId = stream_manager_api_.SendProtobuf(streamName, 0, protobufVec)

    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    key = b'mxpi_tensorinfer0'
    keyVec = StringVector()
    keyVec.push_back(key)
    inferResult = stream_manager_api_.GetProtobuf(streamName, 0, keyVec)
    if inferResult.size() == 0:
        print("inferResult is null")
        exit()
    if inferResult[0].errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d" % (
            inferResult[0].errorCode))
        exit()
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(inferResult[0].messageBuf)
    vision_data_ = result.tensorPackageVec[0].tensorVec[0].dataStr
    vision_data_ = np.frombuffer(vision_data_, dtype=np.float32)
    shape = result.tensorPackageVec[0].tensorVec[0].tensorShape
    vision_data_ = vision_data_.reshape(shape)
    return vision_data_


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
        exit()

    with open(pipeline_path, "r") as file:
        json_str = file.read()
    pipeline = json.loads(json_str)

    pipelineStr = json.dumps(pipeline).encode()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    if len(os.listdir(data_path)) == 0:
        raise RuntimeError("No Input Image!")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for img_name in tqdm(os.listdir(data_path)):
        img_path = os.path.join(data_path, img_name)
        vision_data = infer(img_path, streamManagerApi)
        color = colormap_cityscapes()
        res = np.argmax(vision_data, axis=1)
        res = np.transpose(color[res], (0, 1, 2, 3))
        res = np.squeeze(res, axis=0)
        output_img_path = os.path.join(output_path, img_name)
        cv2.imwrite(output_img_path, res)
    streamManagerApi.DestroyAllStreams()
