# Copyright 2021 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import time
import cv2
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

# The following belongs to the SDK Process
streamManagerApi = StreamManagerApi()
# init stream manager
ret = streamManagerApi.InitManager()
if ret != 0:
    print("Failed to init Stream manager, ret=%s" % str(ret))
    exit()
# Mark start time
start = time.time()
# create streams by pipeline config file
# load  pipline
with open("../pipeline/fire_p.pipeline", 'rb') as f:
    pipelineStr = f.read()
ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
# Print error message
if ret != 0:
    print("Failed to create Stream, ret=%s" % str(ret))
    exit()

# Input object of streams -- detection target
PATH = "./val/"
for item in os.listdir(PATH):
    img_path = os.path.join(PATH,item)
    print("file_path:",img_path)
    img_name = item.split(".")[0]
    img_txt = "./detection-test-result/" + img_name + ".txt"
    if os.path.exists(img_txt):
        os.remove(img_txt)
    dataInput = MxDataInput()
    if os.path.exists(img_path) != 1:
        print("The test image does not exist.")

    with open(img_path, 'rb') as f:
        dataInput.data = f.read()

    streamName = b'detection'
    inPluginId = 0
    # Send data to streams by SendDataWithUniqueId()
    uniqueId = streamManagerApi.SendDataWithUniqueId(streamName, inPluginId, dataInput)

    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    # Get results from streams by GetResultWithUniqueId()
    infer_result = streamManagerApi.GetResultWithUniqueId(streamName, uniqueId, 3000)
    if infer_result.errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
        infer_result.errorCode, infer_result.data.decode()))
        exit()

    # get ObjectList
    results = json.loads(infer_result.data.decode())
    img = cv2.imread(img_path)
    img_shape = img.shape
    bboxes = []
    key = "MxpiObject"
    if key not in results.keys():
        continue
    for bbox in results['MxpiObject']:
        bboxes = {'x0': int(bbox['x0']),
                  'x1': int(bbox['x1']),
                  'y0': int(bbox['y0']),
                  'y1': int(bbox['y1']),
                  'confidence': round(bbox['classVec'][0]['confidence'], 4),
                  'text': bbox['classVec'][0]['className']}
        text = "{}{}".format(str(bboxes['confidence']), " ")
        L1 = []
        L1.append(int(bboxes['x0']))
        L1.append(int(bboxes['x1']))
        L1.append(int(bboxes['y0']))
        L1.append(int(bboxes['y1']))
        L1.append(bboxes['confidence'])
        L1.append(bboxes['text'])

        # save txt for results
        with open(img_txt,"a+") as f:
            content = '{} {} {} {} {} {}'.format(L1[5], L1[4], L1[0], L1[2], L1[1], L1[3])
            f.write(content)
            f.write('\n')

end = time.time()
# Mark spend time
print("Spend time: ", end - start)
# Destroy All Streams
streamManagerApi.DestroyAllStreams()
