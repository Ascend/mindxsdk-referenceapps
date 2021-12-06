
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
    img = '../img/7'   # 修改图片地址
    img_format = '.jpg'
    # 构建pipeline
    with open("../Lane.pipeline", 'rb') as f:
    pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # 构建流的输入对象--检测目标
    img_path = img + img_format
    dataInput = MxDataInput()
    if os.path.exists(img_path) != 1:
        print("The test image does not exist.")
    try :
        with open(img_path, 'rb') as f:
        dataInput.data = f.read()
    except FileNotFoundError:
        print(img_path, "doesn't exist. Exit.")
        exit()
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
    vis = cv2.imread(img_path)
    img_size = vis.shape
    img_w, img_h = 1640, 590  # 模型训练需要输入图片分辨率
    vis_resize = cv2.resize(vis, (img_w,img_h), interpolation=cv2.INTER_CUBIC)
    cls_num_per_lane = 18   # 车道线画点最大值
    row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]  # 模型训练锚点
    col_sample = np.linspace(0, 800 - 1, 200)
    col_sample_w = col_sample[1] - col_sample[0]
    res_path = img + '_r' + img_format
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)   # 坐标点计算
                    cv2.circle(vis, ppp, 5, (0, 255, 0), -1)
    vis_res = cv2.resize(vis_resize, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)       
    cv2.imwrite(res_path, vis_res)
