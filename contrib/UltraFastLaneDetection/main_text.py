
import json
import os
import cv2
import numpy as np
import scipy.special
import tqdm
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    # 新建一个流管理StreamManager对象并初始化
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    img = '../test6.jpg'  # 修改图片地址
    # 构建pipeline
    pipeline = {
        "detection": {
            "stream_config": {
                "deviceId": "0"
            },
            "appsrc0": {
                "props": {
                    "blocksize": "409600"
                },
                "factory": "appsrc",
                "next": "mxpi_imagedecoder0"
            },
            "mxpi_imagedecoder0": {
                "props": {
                    "deviceId": "0"
                },
                "factory": "mxpi_imagedecoder",
                "next": "mxpi_imageresize0"
            },
            "mxpi_imageresize0": {
                "props": {
                    "dataSource": "mxpi_imagedecoder0",
                    "resizeHeight": "288",
                    "resizeWidth": "800"
                },
                "factory": "mxpi_imageresize",
                "next": "mxpi_tensorinfer0"
            },
            "mxpi_tensorinfer0": {
                "props": {
                    "dataSource": "mxpi_imageresize0",
                    "modelPath": "../model/culane_18_2.om"
                },
                "factory": "mxpi_tensorinfer",
                "next": "mxpi_objectpostprocessor0"
            },
            "mxpi_objectpostprocessor0": {
                "props": {
                    "dataSource": "mxpi_tensorinfer0",
                    "postProcessConfigPath": "../model/yolov3_tf_bs1_fp16.cfg",
                    "labelPath": "../model/coco.names",
                    "postProcessLibPath": "../libLanepostprocess.so"
                },
                "factory": "mxpi_objectpostprocessor",
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

    # 构建流的输入对象--检测目标
    dataInput = MxDataInput()
    if os.path.exists(img) != 1:
        print("The test image does not exist.")

    with open(img, 'rb') as f:
        dataInput.data = f.read()
    streamName = b'detection'
    inPluginId = 0
    # 根据流名将检测目标传入流中
    uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)

    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    keys = [b"mxpi_objectpostprocessor0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)
    # # 从流中取出对应插件的输出数
    infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)
    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(infer_result[0].messageBuf)

    out = []
    for results in objectList.objectVec:
        out.append(results.x0)
        out.append(results.y0)
        out.append(results.x1)
        out.append(results.y1)
    out_j = np.array(out).reshape(18,4)

    # #后处理___________________________________________________________________________________________________________________
    img_w, img_h = 1640, 590
    cls_num_per_lane = 18
    row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
    col_sample = np.linspace(0, 800 - 1, 200)
    col_sample_w = col_sample[1] - col_sample[0]

    vis = cv2.imread(img)
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                    cv2.circle(vis, ppp, 5, (0, 255, 0), -1)
                    cv2.imwrite(img, vis)
