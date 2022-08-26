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
import cv2
import numpy as np
from PIL import Image
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector, InProtobufVector, MxProtobufIn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import imageio

def infer(data, streamManagerApi_):
    dataInput = MxDataInput()

    streamName = b'pranet'
    dataInput.data = data
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
    uniqueId = streamManagerApi_.SendProtobuf(streamName, 0, protobufVec)

    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    key = b'mxpi_tensorinfer0'
    keyVec = StringVector()
    keyVec.push_back(key)
    inferResult = streamManagerApi_.GetProtobuf(streamName, 0, keyVec)
    if inferResult.size() == 0:
        print("inferResult is null")
        exit()
    if inferResult[0].errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d" % (
            inferResult[0].errorCode))
        exit()
    
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(inferResult[0].messageBuf)
    vision_data_ = result.tensorPackageVec[0].tensorVec[3].dataStr
    vision_data_ = np.frombuffer(vision_data_, dtype=np.float32)
    shape = result.tensorPackageVec[0].tensorVec[3].tensorShape
    vision_data_ = vision_data_.reshape(shape)
    return vision_data_

def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

if __name__ == '__main__':

    # python infer.py --pipeline_path pranet_pipeline.json --data_path=/home/weigang1/gpf/PraNet/TestDataset/Kvasir/images --output_path ./infer_result

    parser = ArgumentParser()
    parser.add_argument('--pipeline_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type=str)
    config = parser.parse_args()

    result_path = "./"
    pipeline_path = config.pipeline_path
    data_path = config.data_path
    output_path = config.output_path

    images_path = '{}/images/'.format(data_path)
    gts_path = '{}/masks/'.format(data_path)

    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    with open(pipeline_path, "r") as file:
        json_str = file.read()
    pipeline = json.loads(json_str)
    pipelineStr = json.dumps(pipeline).encode()

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])]
    )

    if len(os.listdir(data_path)) == 0:
        raise RuntimeError("No Input Image!")

    for data in tqdm(os.listdir(data_path)):
        image_path = os.path.join(data_path, data)
        image = rgb_loader(image_path)
        shape = image.size
        image = transform(image).unsqueeze(0)
        image = np.array(image).astype(np.float32)
        res = infer(image.tobytes(), streamManagerApi)
        res = np.reshape(res, (1, 1, 352, 352))
        res = torch.from_numpy(res)
        res = F.upsample(res, size=shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        _, name = os.path.split(image_path)
        imageio.imwrite(os.path.join(output_path, name), res)

    streamManagerApi.DestroyAllStreams()

