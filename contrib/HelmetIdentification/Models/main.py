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

"""
func：Call the sdk interface to create a stream, and perform the helmet recognition inference process to obtain the recognition result
"""

# init stream manager
streamManagerApi = StreamManagerApi()
ret = streamManagerApi.InitManager()
if ret != 0:
    print("Failed to init Stream manager, ret=%s" % str(ret))
    
# create streams by pipeline config file

with open("HelmetDetection.pipline", 'rb') as f:
    pipelineStr = f.read()
ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
if ret != 0:
    print("Failed to create Stream, ret=%s" % str(ret))
    

# Obtain the inference result by specifying streamName and keyVec.
streamName = b'Detection'
keyVec0 = StringVector()
keyVec0.push_back(b"ReservedFrameInfo")
keyVec0.push_back(b"mxpi_modelinfer0")
keyVec0.push_back(b"mxpi_motsimplesort0")
keyVec0.push_back(b"mxpi_videodecoder0")
keyVec0.push_back(b"mxpi_videodecoder1")

keyVec1 = StringVector()
keyVec1.push_back(b"ReservedFrameInfo")
keyVec1.push_back(b"mxpi_videodecoder0")
keyVec1.push_back(b"mxpi_videodecoder1")
i = 0

while True:
    t0 = time.time()
    inferResult0 = streamManagerApi.GetProtobuf(streamName, 0, keyVec0)
    if inferResult0[0].errorCode != 0:
        if inferResult0[0].errorCode == 1001:
            print('Object detection result of model infer is null!!!')
        continue

    # add inferennce data into DATA structure
    FrameList0 = MxpiDataType.MxpiFrameInfo()
    FrameList0.ParseFromString(inferResult0[0].messageBuf)
    
    ObjectList = MxpiDataType.MxpiObjectList()
    ObjectList.ParseFromString(inferResult0[1].messageBuf)
    ObjectListData = ObjectList.objectVec

    trackLetList = MxpiDataType.MxpiTrackLetList()
    trackLetList.ParseFromString(inferResult0[2].messageBuf)
    trackLetData = trackLetList.trackLetVec
    
    visionList0 = MxpiDataType.MxpiVisionList()
    visionList0.ParseFromString(inferResult0[3].messageBuf)
    visionData0 = visionList0.visionVec[0].visionData.dataStr
    visionInfo0 = visionList0.visionVec[0].visionInfo

    # cv2：YUV2BGR
    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    img_yuv = np.frombuffer(visionData0, dtype=np.uint8)
    img_yuv = img_yuv.reshape(visionInfo0.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo0.widthAligned)
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

    img1_shape = [640, 640]
    img0_shape = [visionInfo0.heightAligned, visionInfo0.widthAligned]
    print(img0_shape)
    bboxes = []
    color = [random.randint(0, 255) for _ in range(3)]
    tl = (round(0.0002 * (img0.shape[0] + img0.shape[1])) + 0.35)
    tf = max(tl - 1, 1)
    bbox = []
    for bbox in imgLi1:
        if bbox[5] == 'head':
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

            cv2.putText(img0, str(bboxes['confidence']), (L1[0], L1[2]), 0, tl, [225, 255, 255], thickness=tf,
                        lineType=cv2.LINE_AA)  # rectangle tittle color [255,255,255]
            cv2.rectangle(img0, (L1[0], L1[2]), (L1[1], L1[3]), (0, 0, 255), 2) # rectangle color [255,255,255]

            if FrameList0.channelId == 0:
                oringe_imgfile = './output/one/image/image' + str(FrameList0.channelId) + '-' + str(
                    FrameList0.frameId) + '.jpg'
                infer_imgfile = './output/one/inference/image' + str(FrameList0.channelId) + '-' + str(
                    FrameList0.frameId) + '.jpg'
                if os.path.exists(oringe_imgfile):
                    os.remove(oringe_imgfile)
                cv2.imwrite(oringe_imgfile, img0)
                if bboxes['trackid'] is not None and bboxes['age'] == 1:
                    if os.path.exists(infer_imgfile):
                        os.remove(infer_imgfile)
                    cv2.imwrite(infer_imgfile, img0)
            else:
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

        # output inference data ：frame info and img info
        FrameList1 = MxpiDataType.MxpiFrameInfo()
        FrameList1.ParseFromString(inferResult1[0].messageBuf)
        
        visionList1 = MxpiDataType.MxpiVisionList()
        visionList1.ParseFromString(inferResult1[1].messageBuf)
        visionData1 = visionList1.visionVec[0].visionData.dataStr
        visionInfo1 = visionList1.visionVec[0].visionInfo

        # cv2：YUV2BGR
        YUV_BYTES_NU = 3
        YUV_BYTES_DE = 2
        img_yuv = np.frombuffer(visionData1, dtype=np.uint8)
        img_yuv = img_yuv.reshape(visionInfo1.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo1.widthAligned)
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_NV12)

        if FrameList1.channelId == 0:
            oringe_imgfile = './output/one/image/image' + str(FrameList1.channelId) + '-' + str(
                FrameList1.frameId) + '.jpg'
            if os.path.exists(oringe_imgfile):
                os.remove(oringe_imgfile)
            cv2.imwrite(oringe_imgfile, img)
        else:
            oringe_imgfile = './output/two/image/image' + str(FrameList1.channelId) + '-' + str(
                FrameList1.frameId) + '.jpg'
            if os.path.exists(oringe_imgfile):
                os.remove(oringe_imgfile)
            cv2.imwrite(oringe_imgfile, img)

streamManagerApi.DestroyAllStreams()
