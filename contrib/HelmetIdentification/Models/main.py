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

import cv2
import numpy as np
import random
import os
import time
import StreamManagerApi
import MxpiDataType_pb2 as MxpiDataType

# init stream manager
streamManagerApi = StreamManagerApi.StreamManagerApi()
ret = streamManagerApi.InitManager()
if ret != 0:
    print("Failed to init Stream manager, ret=%s" % str(ret))

# create streams by pipeline config file
#load  pipline
with open("HelmetDetection.pipline", 'rb') as f:
    pipelineStr = f.read()
ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
# Print error message
if ret != 0:
    print("Failed to create Stream, ret=%s" % str(ret))


# Obtain the inference result by specifying streamName and keyVec
# The data that needs to be obtained is searched by the plug-in name
# Stream name
streamName = b'Detection'
keyVec0 = StreamManagerApi.StringVector()
keyVec0.push_back(b"ReservedFrameInfo")
keyVec0.push_back(b"mxpi_modelinfer0")
keyVec0.push_back(b"mxpi_motsimplesort0")
keyVec0.push_back(b"mxpi_videodecoder0")
keyVec0.push_back(b"mxpi_videodecoder1")
# Output data required by the second output
keyVec1 = StreamManagerApi.StringVector()
keyVec1.push_back(b"ReservedFrameInfo")
keyVec1.push_back(b"mxpi_videodecoder0")
keyVec1.push_back(b"mxpi_videodecoder1")
i = 0

while True:
    t0 = time.time()
    # Get data through GetProtobuf interface
    inferResult0 = streamManagerApi.GetProtobuf(streamName, 0, keyVec0)
    # output errorCode
    if inferResult0[0].errorCode != 0:
        # Print error message
        if inferResult0[0].errorCode == 1001:
            print('Object detection result of model infer is null!!!')
        continue

    # add inferennce data into DATA structure
    # Frame information structure
    FrameList0 = MxpiDataType.MxpiFrameInfo()
    FrameList0.ParseFromString(inferResult0[0].messageBuf)
    # Target object structure
    ObjectList = MxpiDataType.MxpiObjectList()
    ObjectList.ParseFromString(inferResult0[1].messageBuf)
    # Get target box information
    ObjectListData = ObjectList.objectVec
    # track structure
    trackLetList = MxpiDataType.MxpiTrackLetList()
    trackLetList.ParseFromString(inferResult0[2].messageBuf)
    # Obtain tracking information
    trackLetData = trackLetList.trackLetVec
    # image structure
    visionList0 = MxpiDataType.MxpiVisionList()
    visionList0.ParseFromString(inferResult0[3].messageBuf)
    visionData0 = visionList0.visionVec[0].visionData.dataStr
    # Get picture information
    visionInfo0 = visionList0.visionVec[0].visionInfo

    # cv2:YUV2BGR
    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    img_yuv = np.frombuffer(visionData0, dtype=np.uint8)
    # reshape
    img_yuv = img_yuv.reshape(visionInfo0.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo0.widthAligned)
    # Color gamut conversion
    img0 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_NV12)
    t1 = time.time()
    # put inference into dict,
    imgLi1 = []
    for k in range(len(ObjectList.objectVec)):
        imgLi = [round(ObjectListData[k].x0, 4), round(ObjectListData[k].x1, 4), round(ObjectListData[k].y0, 4),
                 round(ObjectListData[k].y1, 4),
                 round(ObjectListData[k].classVec[0].confidence, 4), ObjectListData[k].classVec[0].className,
                 trackLetData[k].trackId, trackLetData[k].age]
        imgLi1.append(imgLi)

    # img1_shape is the size after padding
    # img0_shape is the original image size
    img1_shape = [640, 640]
    img0_shape = [visionInfo0.heightAligned, visionInfo0.widthAligned]
    print(img0_shape)
    bboxes = []
    # The color and title of the rectangle
    color = [random.randint(0, 255) for _ in range(3)]
    tl = (round(0.0002 * (img0.shape[0] + img0.shape[1])) + 0.35)
    tf = max(tl - 1, 1)
    bbox = []
    for bbox in imgLi1:
        if bbox[5] == 'head':
        # Determine whether it is helmet
            bboxes = {'x0': int(bbox[0]),
                      'x1': int(bbox[1]),
                      'y0': int(bbox[2]),
                      'y1': int(bbox[3]),
                      'confidence': round(bbox[4], 4),
                      'trackid': int(bbox[6]),
                      'age': int(bbox[7])
                      }
            print(bboxes)
            L1 = []
            L1.append(int(bboxes['x0']))
            L1.append(int(bboxes['x1']))
            L1.append(int(bboxes['y0']))
            L1.append(int(bboxes['y1']))
            L1 = np.array(L1, dtype=np.int32)
            # Draw rectangle
            cv2.putText(img0, str(bboxes['confidence']), (L1[0], L1[2]), 0, tl, [225, 255, 255], thickness=tf,
                        lineType=cv2.LINE_AA)
            # rectangle color [255,255,255]
            cv2.rectangle(img0, (L1[0], L1[2]), (L1[1], L1[3]), (0, 0, 255), 2)
            if bboxes['trackid'] is not None and bboxes['age'] == 1:
                cv2.imwrite(infer_imgfile, img0)

    # Save pictures in two ways
    if FrameList0.channelId == 0:
        oringe_imgfile = './output/one/image/image' + str(FrameList0.channelId) + '-' + str(
            FrameList0.frameId) + '.jpg'
        # Inference result save path
        infer_imgfile = './output/one/inference/image' + str(FrameList0.channelId) + '-' + str(
            FrameList0.frameId) + '.jpg'
        # Warning result save path
        if os.path.exists(oringe_imgfile):
            os.remove(oringe_imgfile)
        cv2.imwrite(oringe_imgfile, img0)
        # If age is 1, then output this picture to inference
        if bboxes['trackid'] is not None and bboxes['age'] == 1:
            if os.path.exists(infer_imgfile):
                os.remove(infer_imgfile)
            cv2.imwrite(infer_imgfile, img0)
    else:
        # when channelId equal 1
        oringe_imgfile = './output/two/image/image' + str(FrameList0.channelId) + '-' + str(
            FrameList0.frameId) + '.jpg'
        infer_imgfile = './output/two/inference/image' + str(FrameList0.channelId) + '-' + str(
            FrameList0.frameId) + '.jpg'
        if os.path.exists(oringe_imgfile):
            os.remove(oringe_imgfile)
        cv2.imwrite(oringe_imgfile, img0)
        if bboxes['trackid'] is not None and bboxes['age'] == 1:
            if os.path.exists(infer_imgfile):
                os.remove(infer_imgfile)
            cv2.imwrite(infer_imgfile, img0)

    # output 6 frame info per inference
    for i in range(6):
        inferResult1 = streamManagerApi.GetProtobuf(streamName, 1, keyVec1)
        if inferResult1[0].errorCode != 0:
            print("GetProtobuf error. errorCode0=%d" % (inferResult1[0].errorCode))
            break

        # output inference data:frame info and img info
        # Frame information structure
        FrameList1 = MxpiDataType.MxpiFrameInfo()
        # Get channel id and frame id
        FrameList1.ParseFromString(inferResult1[0].messageBuf)
        # image structure
        visionList1 = MxpiDataType.MxpiVisionList()
        visionList1.ParseFromString(inferResult1[1].messageBuf)
        visionData1 = visionList1.visionVec[0].visionData.dataStr
        # Get picture information
        visionInfo1 = visionList1.visionVec[0].visionInfo

        # cv2:YUV2BGR
        YUV_BYTES_NU = 3
        YUV_BYTES_DE = 2
        img_yuv = np.frombuffer(visionData1, dtype=np.uint8)
        # reshape
        img_yuv = img_yuv.reshape(visionInfo1.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo1.widthAligned)
        # Color gamut conversion
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_NV12)
        # Save pictures in two ways
        if FrameList1.channelId == 0:
            oringe_imgfile = './output/one/image/image' + str(FrameList1.channelId) + '-' + str(
                FrameList1.frameId) + '.jpg'
            if os.path.exists(oringe_imgfile):
                os.remove(oringe_imgfile)
            cv2.imwrite(oringe_imgfile, img)
        else:
            # when channelId equal 1
            oringe_imgfile = './output/two/image/image' + str(FrameList1.channelId) + '-' + str(
                FrameList1.frameId) + '.jpg'
            if os.path.exists(oringe_imgfile):
                os.remove(oringe_imgfile)
            cv2.imwrite(oringe_imgfile, img)
# Destroy All Streams
streamManagerApi.DestroyAllStreams()
