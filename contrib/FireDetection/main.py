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
import cv2
import numpy as np

from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import MxpiDataType_pb2 as MxpiDataType

# The following belongs to the SDK Process
streamManagerApi = StreamManagerApi()
# init stream manager
ret = streamManagerApi.InitManager()
if ret != 0:
    print("Failed to init Stream manager, ret=%s" % str(ret))
    exit()

# create streams by pipeline config file
# load pipline
with open("./pipeline/fire_v.pipeline", 'rb') as f:
    pipelineStr = f.read()
ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
# Print error message
if ret != 0:
    print("Failed to create Stream, ret=%s" % str(ret))

# Stream name
streamName = b'detection'
# Obtain the inference result by specifying streamName and keyVec
# The data that needs to be obtained is searched by the plug-in name
keys = [b"ReservedFrameInfo", b"mxpi_modelinfer0", b"mxpi_videodecoder0"]
keyVec = StringVector()
for key in keys:
    keyVec.push_back(key)

while True:
    # Get data through GetResult
    infer_result = streamManagerApi.GetResult(streamName, b'appsink0', keyVec)

    # Determine whether the output is empty
    if infer_result.metadataVec.size() == 0:
        print("infer_result is null")
        continue

    # Frame information structure
    frameList = MxpiDataType.MxpiFrameInfo()
    frameList.ParseFromString(infer_result.metadataVec[0].serializedMetadata)

    # Objectpostprocessor information
    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(infer_result.metadataVec[1].serializedMetadata)

    # Videodecoder information
    visionList = MxpiDataType.MxpiVisionList()
    visionList.ParseFromString(infer_result.metadataVec[2].serializedMetadata)

    vision_data = visionList.visionVec[0].visionData.dataStr
    visionInfo = visionList.visionVec[0].visionInfo

    # cv2 func YUV to BGR
    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    img_yuv = np.frombuffer(vision_data, np.uint8)
    # reshape
    img_bgr = img_yuv.reshape(visionInfo.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo.widthAligned)
    # Color gamut conversion
    img = cv2.cvtColor(img_bgr, getattr(cv2, "COLOR_YUV2BGR_NV12"))

    bboxes = []
    # fire or not
    if len(objectList.objectVec) == 0:
        continue
    for i in range(len(objectList.objectVec)):
        # get ObjectList
        results = objectList.objectVec[i]
        bboxes = {'x0': int(results.x0),
                 'x1': int(results.x1),
                 'y0': int(results.y0),
                 'y1': int(results.y1),
                 'confidence': round(results.classVec[0].confidence, 4),
                 'text': results.classVec[0].className}

        text = "{}{}".format(str(bboxes['confidence']), " ")

        # Draw rectangle
        for item in bboxes['text']:
            text += item
        cv2.putText(img, text, (bboxes['x0'] + 10, bboxes['y0'] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 1)
        cv2.rectangle(img, (bboxes['x0'], bboxes['y0']), (bboxes['x1'], bboxes['y1']), (255, 0, 0), 2)

    # save picture
    Id = frameList.frameId
    result_path = "./result/"
    if os.path.exists(result_path) != 1:
        os.makedirs("./result/")
    oringe_imgfile = './result/image' + '-' + str(Id) + '.jpg'
    print("Warning! Fire or smoke detected")
    print("Result save in ",oringe_imgfile)
    cv2.imwrite(oringe_imgfile, img)

# Destroy All Streams
streamManagerApi.DestroyAllStreams()
