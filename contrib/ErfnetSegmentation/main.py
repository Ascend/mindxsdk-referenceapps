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
import PIL
from PIL import Image
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector, \
    InProtobufVector, MxProtobufIn
import cv2
from tqdm import tqdm


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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pipeline_path', type=str,
                        default="./pipeline/erfnet_pipeline.pipeline")
    parser.add_argument('--data_path', type=str)
    config = parser.parse_args()

    pipeline_path = config.pipeline_path
    data_path = config.data_path

    INFER_RESULT = "infer_result/"
    if not os.path.exists(INFER_RESULT):
        os.mkdir(INFER_RESULT)

    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        sys.exit()

    with open(pipeline_path, 'rb') as f:
        json_str = f.read()

    pipeline = json.loads(json_str)

    pipelineStr = json.dumps(pipeline).encode()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        sys.exit()

    if len(os.listdir(data_path)) == 0:
        raise RuntimeError("No Input Image!")

    for index, img_name in tqdm(enumerate(os.listdir(data_path))):
        img_path = os.path.join(data_path, img_name)
        print(index, img_path)
        infer(img_path, streamManagerApi)
        RESIMAGE = INFER_RESULT + str(index) + ".png"
        while True:  # 轮询, 等待异步线程
            try:
                preds = Image.open(RESIMAGE).convert('RGB')
                break
            except (OSError, FileNotFoundError, PIL.UnidentifiedImageError, SyntaxError):
                continue
        os.rename(RESIMAGE, os.path.join(INFER_RESULT, "result_"+img_name))
    streamManagerApi.DestroyAllStreams()
