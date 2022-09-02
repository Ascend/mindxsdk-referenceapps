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
import cv2
from tqdm import tqdm


def get_image_binary(img_path_):
    fd_1 = os.open(img_path_, os.O_RDWR | os.O_CREAT)
    file__ = os.fdopen(fd_1, 'rb')
    data = file__.read()
    image = Image.open(file__)
    output = BytesIO()
    image.save(output, format='JPEG')
    return output


def infer(img_path_, stream_manager_api_):
    data_input = MxDataInput()
    data_input.data = get_image_binary(img_path_).getvalue()
    unique_id = stream_manager_api_.SendData(b'erfnet', 0, data_input)
    if unique_id < 0:
        print("Failed to send data to stream.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pipeline_path', type=str,
                        default="./pipeline/erfnet_pipeline.json")
    parser.add_argument('--data_path', type=str)
    config = parser.parse_args()

    pipeline_path = config.pipeline_path
    data_path = config.data_path

    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        sys.exit()

    fd = os.open(pipeline_path, os.O_RDWR | os.O_CREAT)
    file = os.fdopen(fd, 'rb')
    json_str = file.read()
    file.close()

    pipeline = json.loads(json_str)

    pipelineStr = json.dumps(pipeline).encode()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        sys.exit()

    if len(os.listdir(data_path)) == 0:
        raise RuntimeError("No Input Image!")

    for img_name in tqdm(os.listdir(data_path)):
        img_path = os.path.join(data_path, img_name)
        infer(img_path, streamManagerApi)
    streamManagerApi.DestroyAllStreams()
