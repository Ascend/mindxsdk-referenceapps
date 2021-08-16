"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import os
import cv2
import numpy as np
import random
from StreamManagerApi import *

if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    pipeline = {
            "detection": {
                "stream_config": {
                    "deviceId": "1"
                },
                "appsrc0": {
                    "props": {
                        "blocksize": "409600"
                    },
                    "factory": "appsrc",
                    "next": "mxpi_imagedecoder0"
                },
                "mxpi_imagedecoder0": {
                    "factory": "mxpi_imagedecoder",
                    "next": "mxpi_imageresize0"
                },
                "mxpi_imageresize0": {
                    "factory": "mxpi_imageresize",
                    "props": {
                        "dataSource": "mxpi_imagedecoder0",
                        "resizeType": "Resizer_KeepAspectRatio_Fit",
                        "resizeHeight": "640",
                        "resizeWidth": "640"
                    },
                    "next": "queue0"
                },
                "queue0": {
                    "props": {
                        "max-size-buffers": "500"
                    },
                    "factory": "queue",
                    "next": "mxpi_modelinfer0"
                },
                "mxpi_modelinfer0": {
                    "props": {
                        "dataSource": "mxpi_imageresize0",
                        "modelPath": "./ExampleProject/HelmetIdentification/Models/YOLOv5_s.om",
                        "postProcessConfigPath": "./HelmetExample/MAP/VOC2028/Helmet_yolov5.cfg",
                        "labelPath": "./ExampleProject/HelmetIdentification/Models/imgclass.names",
                        "postProcessLibPath": "./MindX_SDK/mxVision/lib/libMpYOLOv5PostProcessor.so"
                    },
                    "factory": "mxpi_modelinfer",
                    "next": "mxpi_dataserialize0"
                },
                "mxpi_dataserialize0": {
                    "props": {
                        "outputDataKeys": "mxpi_modelinfer0"
                    },
                    "factory": "mxpi_dataserialize",
                    "next": "appsink0"
                },
                "appsink0": {
                    "props": {
                        "blocksize": "4096000"
                    },
                    "factory": "appsink"
                }
            }
    }

    pipelineStr = json.dumps(pipeline).encode()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    path = "./TestImages/"
    for file in os.listdir(path):
        img_path = os.path.join(path,file)
        print("file_path:",img_path)
        img_name = file.split(".")[0]
        img_txt = "./detection-test-result/" + img_name + ".txt"
        if os.path.exists(img_txt):
            os.remove(img_txt)
        dataInput = MxDataInput()
        if os.path.exists(img_path) != 1:
            print("The test image does not exist.")

        with open(img_path, 'rb') as f:
            dataInput.data = f.read()


        # Inputs data to a specified stream based on streamName.
        streamName = b'detection'
        ret = streamManagerApi.SendData(streamName, 0, dataInput)

        if ret < 0:
            print("Failed to send data to stream.")
            exit()

        # Obtain the inference result by specifying streamName and uniqueId.
        infer_result = streamManagerApi.GetResult(streamName, 0)
        if infer_result.errorCode != 0:
            print("GetResult error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            exit()

        # print the infer result
        # print(infer_result.data.decode())

        results = json.loads(infer_result.data.decode())
        img1_shape=[640,640]
        img = cv2.imread(img_path)
        img_shape=img.shape
        print(img_shape)
        bboxes = []
        color = [random.randint(0, 255) for _ in range(3)]
        tl=round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        tf= max(tl - 1, 1)
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
            print(bboxes)
            L1 = []
            L1.append(int(bboxes['x0']))
            L1.append(int(bboxes['x1']))
            L1.append(int(bboxes['y0']))
            L1.append(int(bboxes['y1']))
            # L1=np.array(L1,dtype=np.int32)
            L1.append(bboxes['confidence'])
            L1.append(bboxes['text'])
            print(L1)

            with open(img_txt,"a+") as f:
                    content = '{} {} {} {} {} {}'.format(L1[5], L1[4], L1[0], L1[2], L1[1], L1[3])
                    f.write(content)
                    f.write('\n')

    # destroy streams
    streamManagerApi.DestroyAllStreams()