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
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector, \
    InProtobufVector, MxProtobufIn
from tqdm import tqdm
import imageio


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


def infer(img_path_, stream_manager_api):
    data_input = MxDataInput()
    with open(img_path_, 'rb') as file__:
        image = Image.open(file__)
        output = BytesIO()
        image.save(output, format='JPEG')
        data_input.data = output.getvalue()
    unique_id = stream_manager_api.SendData(b'pranet', 0, data_input)
    if unique_id < 0:
        print("Failed to send data to stream.")


def rgb_loader(path):
    with open(path, 'rb') as file_:
        img = Image.open(file_)
        return img.convert('RGB')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--pipeline_path', type=str,
                        default="pipeline/pranet_pipeline.json")
    parser.add_argument('--data_path', type=str)
    config = parser.parse_args()

    pipeline_path = config.pipeline_path
    data_path = config.data_path

    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        sys.exit()

    with open(pipeline_path, "r") as file:
        json_str = file.read()
    pipeline = json.loads(json_str)
    pipelineStr = json.dumps(pipeline).encode()

    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        sys.exit()

    mean = np.array([[[0.485]], [[0.456]], [[0.406]]], dtype=np.float32)
    std = np.array([[[0.229]], [[0.224]], [[0.225]]], dtype=np.float32)

    if len(os.listdir(data_path)) == 0:
        raise RuntimeError("No Input Image!")

    for index, data in tqdm(enumerate(os.listdir(data_path))):
        image_path = os.path.join(data_path, data)

        print(image_path)
        infer(image_path, streamManagerApi)

        while True:  # 轮询, 等待异步线程
            if os.path.exists("infer_result/" + str(index) + ".png"):
                break
    streamManagerApi.DestroyAllStreams()
