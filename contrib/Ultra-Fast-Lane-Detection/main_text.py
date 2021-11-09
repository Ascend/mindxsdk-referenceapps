
import json
import os
import cv2
import numpy as np
import scipy.special, tqdm
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    # 新建一个流管理StreamManager对象并初始化
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

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
                    "modelPath": "/home/zhongzhi3/MindX_SDK/mxVision/samples/mxVision/Wmm/python/Lane-Detection-master/model/culane_18_2.om"
                },
                "factory": "mxpi_tensorinfer",
                "next": "mxpi_objectpostprocessor0"
            },
            "mxpi_objectpostprocessor0": {
                "props": {
                    "dataSource": "mxpi_tensorinfer0",
                    "postProcessConfigPath": "/home/zhongzhi3/MindX_SDK/mxVision/samples/mxVision/Wmm/python/Lane-Detection-master/model/yolov3_tf_bs1_fp16.cfg",
                    "labelPath": "/home/zhongzhi3/MindX_SDK/mxVision/samples/mxVision/Wmm/python/Lane-Detection-master/model/coco.names",
                    "postProcessLibPath": "/home/zhongzhi3/MindX_SDK/mxVision/samples/mxVision/SamplePostProcess/libyolov3postprocess.so"
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
    if os.path.exists('test6.jpg') != 1:
        print("The test image does not exist.")

    with open("test6.jpg", 'rb') as f:
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

    # if infer_result.size() == 0:
    #     print("infer_result is null")
    #     exit()

    # if infer_result[0].errorCode != 0:
    #     print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
    #         infer_result[0].errorCode, infer_result[0].data.decode()))
    #     exit()

    # #___________添加
    # #print(infer_result[0].messageBuf)
    # object_list = MxpiDataType.MxpiTensorPackageList()
    # object_list.ParseFromString(infer_result[0].messageBuf)
    # print(object_list)
    #mxpi_objectpostprocessor0模型后处理插件输出信息
    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(infer_result[0].messageBuf)

    out = []
    for results in objectList.objectVec:
        out.append(results.x0)
        out.append(results.y0)
        out.append(results.x1)
        out.append(results.y1)

    out_j = np.array(out).reshape(18,4)
    print("hhhhhhhh--------", out_j)
    # bboxes = []
    # bboxes = {'x0': int(results.x0),
    # 'y0': int(results.y0),
    # 'x1': int(results.x1),
    # 'y1': int(results.y1),
    # 'confidence': round(results.classVec[0].confidence, 4),
    # 'text': results.classVec[0].className}
    # print(bboxes)
    #a=np.frombuffer(object_list.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
    # out=a.reshape(201,18, 4)
    #print(a)
    #
    # #后处理___________________________________________________________________________________________________________________
    img_w, img_h = 1640, 590
    cls_num_per_lane = 18
    row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
    # #row_anchor = [121,126, 131,136, 141,146, 150,155, 160,165, 170,175, 180,185, 189,194, 199,204, 209,214, 219, 224,228,233, 238,243, 248,253, 258,263, 267,272, 277,282, 287]
    col_sample = np.linspace(0, 800 - 1, 200)
    col_sample_w = col_sample[1] - col_sample[0]
    # #print(out)
    #
    # out_j = out[:, ::-1, :]  #中心对称转换顺序
    # #print(out_j)
    # out_k = out_j[:-1, :, :]  #去除最后一组数据 输出数据为200*18*4
    # print(out_k)
    # prob = scipy.special.softmax(out_k, axis=0)#（200.18.4）200组18*4数据进行softmax计算（列计算）
    # #print(prob.shape)
    # idx = np.arange(200) + 1#（200.1.1）1...200
    #
    # idx = idx.reshape(-1, 1, 1)#变成列
    # #print(idx.shape)
    # m = prob * idx
    # #print(m)
    # loc = np.sum(m, axis=0)     #求和200组对应位置数据和18*4
    # #print(loc.shape)
    # out_j = np.argmax(out_j, axis=0)  #找出对应数据最大值所在组数   18*4
    # #print(out_j)
    # loc[out_j == 200] = 0   #让最大组数在数组中为0
    # out_j = loc
    #
    #     # import pdb; pdb.set_trace()
    vis =cv2.imread(r'/home/zhongzhi3/MindX_SDK/mxVision/samples/mxVision/Wmm/python/Lane-Detection-master/test6.jpg')
    for i in range(out_j.shape[1]):#4条线18*4中的4
    #for i in range(4):
        if np.sum(out_j[:, i] != 0) > 2:#18个点数据和大于2
            for k in range(out_j.shape[0]):   #画点
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                    cv2.circle(vis, ppp, 5, (0, 255, 0), -1)
                    cv2.imwrite(r'/home/zhongzhi3/MindX_SDK/mxVision/samples/mxVision/Wmm/python/Lane-Detection-master/img/'  + '6-3.jpg', vis)
     #修改————————————————————

    # Get target box information
    #objectlist_data = object_list.objectVec
    #添加

    #YUV_BYTES_NU = 3
    #YUV_BYTES_DE = 2
    # mxpi_objectpostprocessor0模型后处理插件输出信息
    #objectList = MxpiDataType.MxpiObjectList()
    #objectList.ParseFromString(infer_result[0].messageBuf)
    #print(objectList)

    # mxpi_imagedecoder0 图像解码插件输出信息
    #visionList = MxpiDataType.MxpiVisionList()
    #visionList.ParseFromString(infer_result[1].messageBuf)

    #vision_data = visionList.visionVec[0].visionData.dataStr
    #visionInfo = visionList.visionVec[0].visionInfo

    # 用输出原件信息初始化OpenCV图像信息矩阵
    #img_yuv = np.frombuffer(vision_data, np.uint8)

    #img_bgr = img_yuv.reshape(visionInfo.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo.widthAligned)
    #img = cv2.cvtColor(img_bgr, getattr(cv2, "COLOR_YUV2BGR_NV12"))

    # 打印出推理结果
    #results = objectList.objectVec[0]

    #bboxes = []
    #bboxes = {'x0': int(results.x0),
              #'y0': int(results.y0),
              #'y1': int(results.y1),
              #'confidence': round(results.classVec[0].confidence, 4),
              #'text': results.classVec[0].className}

    #text = "{}{}".format(str(bboxes['confidence']), " ")

    #for item in bboxes['text']:
        #text += item
    #cv2.putText(img, text, (bboxes['x0'] + 10, bboxes['y0'] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 1)
    #cv2.rectangle(img, (bboxes['x0'], bboxes['y0']), (bboxes['x1'], bboxes['y1']), (255, 0, 0), 2)

    #cv2.imwrite("./result.jpg", img)

    # destroy streams

    #treamManagerApi.DestroyAllStreams()