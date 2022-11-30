# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import numpy as np

from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import MxpiDataType_pb2 as MxpiDataType


class ostream:
    def __init__(self, file):
        self.file = file

    def __lshift__(self, obj):
        self.file.write(str(obj))
        return self


cout = ostream(sys.stdout)
endl = '/n'

# The following belongs to the SDK Process
streamManagerApi = StreamManagerApi()
# Init stream manager
ret = streamManagerApi.InitManager()
if ret != 0:
    cout << 'Failed to init Stream manager, ret=' << str(ret) << endl
    exit()
# Mark start time
# start = time.time()
# Create streams by pipeline config file
# load  pipline
with open("../pipeline/burpee_detection_p.pipeline", 'rb') as f:
    pipelineStr = f.read()
ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
# Print error message
if ret != 0:
    cout << 'Failed to create Stream, ret=' << str(ret) << endl
    exit()
# Stream name
STREAM_NAME = b'detection'
# Obtain the inference result by specifying streamName and keyVec
# The data that needs to be obtained is searched by the plug-in name
keys = [b"ReservedFrameInfo", b"mxpi_modelinfer0", b"mxpi_videodecoder0"]
keyVec = StringVector()
for key in keys:
    keyVec.push_back(key)

STATE = 0
ACTION_CNT = 0
# Config the output video 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video_result.mp4', fourcc, 30, (1280, 720))

while True:

    # Get data through GetResult   
    infer_result = streamManagerApi.GetResult(STREAM_NAME, b'appsink0', keyVec)

    # Determine whether the output is empty
    if infer_result.metadataVec.size() == 0:
        cout << 'infer_result is null' << endl
        continue

    # Frame information structure
    frameList = MxpiDataType.MxpiFrameInfo()
    frameList.ParseFromString(infer_result.metadataVec[0].serializedMetadata)

    # Object postprocessor information
    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(infer_result.metadataVec[1].serializedMetadata)

    # Video-decoder information
    visionList = MxpiDataType.MxpiVisionList()
    visionList.ParseFromString(infer_result.metadataVec[2].serializedMetadata)
    vision_data = visionList.visionVec[0].visionData.dataStr
    visionInfo = visionList.visionVec[0].visionInfo

    # cv2 func YUV to BGR
    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    img_yuv = np.frombuffer(vision_data, np.uint8)
    # Reshape
    img_bgr = img_yuv.reshape(visionInfo.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE,
                              visionInfo.widthAligned)
    # Color gamut conversion
    img = cv2.cvtColor(img_bgr, getattr(cv2, "COLOR_YUV2BGR_NV12"))

    BEST_CONFIDENCE = 0

    if len(objectList.objectVec) == 0:
        continue

    for i in range(len(objectList.objectVec)):
        # Get ObjectList
        results = objectList.objectVec[i]
        # Get the confidence
        confidence = round(results.classVec[0].confidence, 4)
        # Save the best confidence and its information
        if confidence > BEST_CONFIDENCE:
            BEST_CONFIDENCE = confidence
            best_bboxes = {'x0': int(results.x0),
                           'x1': int(results.x1),
                           'y0': int(results.y0),
                           'y1': int(results.y1),
                           'text': results.classVec[0].className}
            action = best_bboxes['text']
            TEXT = "{}{}".format(str(BEST_CONFIDENCE), " ")

    # Draw rectangle and txt for visualization
    for item in best_bboxes['text']:
        TEXT += item
    cv2.putText(img, TEXT, (best_bboxes['x0'] + 10, best_bboxes['y0'] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 1)
    cv2.rectangle(img, (best_bboxes['x0'], best_bboxes['y0']), (best_bboxes['x1'], best_bboxes['y1']),
                  (255, 0, 0), 2)

    # State change         
    if STATE == 0:
        if action == "crouch":
            STATE = 1
    elif STATE == 1:
        if action == "support":
            STATE = 2
    elif STATE == 2:
        if action == "crouch":
            STATE = 3
    elif STATE == 3:
        if action == "jump":
            STATE = 0
            ACTION_CNT = ACTION_CNT + 1

    # Save txt for results
    FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if os.path.exists("result.txt"):
        os.remove("result.txt")
    with os.fdopen(os.open('result.txt', FLAGS, 0o755), 'w') as f:
        f.write(str(ACTION_CNT))

    # Save picture
    Id = frameList.frameId
    RESULT_PIC_PATH = "./result_pic/"
    if os.path.exists(RESULT_PIC_PATH) != 1:
        os.makedirs("./result_pic/")
    origin_img_file = './result_pic/image' + '-' + str(Id) + '.jpg'
    cv2.imwrite(origin_img_file, img)

    # Write the video
    out.write(img)

    # Stop detection when it is the lase frame 
    # Or when the frame id comes to be the number you set
    if frameList.isEos or Id > 63:
        out.release()
        break

# Destroy All Streams
streamManagerApi.DestroyAllStreams()
