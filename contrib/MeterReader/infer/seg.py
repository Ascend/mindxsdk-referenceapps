#!/usr/bin/env python
# coding=utf-8

# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

import json
import os
import stat
import cv2
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import sys
import getopt

cur_path = os.path.abspath(os.path.dirname(__file__))
father_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
pipeline_path = os.path.join(father_path, 'pipeline', 'deeplabv3', 'seg.pipeline').replace('\\', '/')
model_path = os.path.join(father_path, 'models', 'deeplabv3', 'seg.om').replace('\\', '/')
postProcessConfigPath = os.path.join(father_path, 'pipeline', 'deeplabv3', 'deeplabv3.cfg').replace('\\', '/')
labelPath = os.path.join(father_path, 'pipeline', 'deeplabv3', 'deeplabv3.names').replace('\\', '/')

def Miou(img,img_pred):
    """
    img: 已标注的原图片
    img_pred: 预测出的图片
    """
    if (img.shape != img_pred.shape):
        print("两个图片形状不一致")
        return

    unique_item_list = np.unique(img) #不同取值的List
    unique_item_dict = {} #不同取值对应的下标dict
    for index in range(len(unique_item_list)):
        item = unique_item_list[index]
        unique_item_dict[item] = index
    num = len(np.unique(unique_item_list)) #共有num个不同取值

    #混淆矩阵
    M = np.zeros((num+1,num+1)) #多加一行一列，用于计算总和
    #统计个数
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            M_i = unique_item_dict[img[i][j]]
            M_j = unique_item_dict[img_pred[i][j]]
            M[M_i][M_j] += 1
    print(M)
    #前num行相加，存在num+1列 【实际下标-1】
    M[:num,num] = np.sum(M[:num,:num],axis = 1)
    #前num+1列相加，放在num+1行【实际下标-1】
    M[num,:num+1] = np.sum(M[:num,:num+1],axis = 0)
    # print(M)

    #计算Miou值
    miou = 0 
    for i in range(num):
        miou += (M[i][i])/(M[i][num] + M[num][i] - M[i][i])
    miou /= num
    print(miou)
    return miou

def get_args():
    argv = sys.argv[1:]
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    print('输入的文件为：', inputfile)
    print('输出的文件为：', outputfile)
    return inputfile,outputfile

if __name__ == '__main__':

    
    # python seg.py --ifile /home/wangyi4/tmp/221021_xhr/images/seg_test_img/110.jpg --ofile /home/wangyi4/tmp/221021_xhr/images/det_rect/
    FILENAME,RESULTFILE = get_args()

    #改写pipeline里面的model路径
    
    file_object = open(pipeline_path,'r')

    content = json.load(file_object)
    modelPath = model_path
    content['seg']['mxpi_tensorinfer0']['props']['modelPath'] = modelPath
    content['seg']['mxpi_semanticsegpostprocessor0']['props']['postProcessConfigPath'] = postProcessConfigPath
    content['seg']['mxpi_semanticsegpostprocessor0']['props']['labelPath'] = labelPath

    # print(content)
    with open(pipeline_path,"w") as f:
        json.dump(content,f)

    steammanager_api = StreamManagerApi()
    # init stream manager
    ret = steammanager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    MODES = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(pipeline_path, os.O_RDONLY, MODES), 'rb') as f:
        pipeline_str = f.read()
    ret = steammanager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    dataInput = MxDataInput()
    # It is best to use absolute path
    # FILENAME = "/home/wangyi4/tmp/221021_xhr/images/seg_test_img/110.jpg"
    if os.path.exists(FILENAME) != 1:
        print("The test image does not exist. Exit.")
        exit()
    with os.fdopen(os.open(FILENAME, os.O_RDONLY, MODES), 'rb') as f:
        dataInput.data = f.read()
    STEAMNAME = b'seg'
    INPLUGINID = 0
    uniqueId = steammanager_api.SendData(STEAMNAME, INPLUGINID, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()
    keys = [b"mxpi_process3"]
    # keys = [b"mxpi_objectpostprocessor0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)
    # ### Test

    infer = steammanager_api.GetResult(STEAMNAME, b'appsink0', keyVec)
    print("-------------------------")
    result = MxpiDataType.MxpiClass()
    result.ParseFromString(infer.metadataVec[0].serializedMetadata)
    print(f"seg_ans {result.confidence}")
    # if(infer.metadataVec.size() == 0):
    #     print("Get no data from stream !")
    #     exit()
    # print("result.metadata size: ", infer.metadataVec.size())
    # infer_result = infer.metadataVec[0]
    # if infer_result.errorCode != 0:
    #     print("GetResult error. errorCode=%d , errMsg=%s" % (infer_result.errorCode, infer_result.errMsg))
    #     exit()
    # result = MxpiDataType.MxpiImageMaskList()
    # result.ParseFromString(infer_result.serializedMetadata)
    # print("imageMaskVec[0].shape: ")
    # print(result.imageMaskVec[0].shape)
    # print("\nimageMaskVec[0].className: ")
    # print(result.imageMaskVec[0].className)
    # pred = np.frombuffer(result.imageMaskVec[0].dataStr
    #                       , dtype=np.uint8)
    # print(pred.shape)
    # print(np.nonzero(pred))
    # print(set(list(pred)))


    # WIDTH = 512
    # HEIGHT = 512
    # CLASS = 1
    # pred = pred * int(255 / pred.max())
    # print(set(list(pred)))
    # img_pred = pred.reshape((HEIGHT, WIDTH))

    # cv2.imwrite("./mask.png" , img_pred)


    # # #######

    # MODES = stat.S_IWUSR | stat.S_IRUSR
    # # det_val_dir = "/home/wangyi4/tmp/seg/det_val"
    # # files= os.listdir(det_val_dir)
    # # det_val_dict = {}
    # # for file in files:
    # # FILENAME = det_val_dir + os.path.sep + file
    # FILENAME = "/home/wangyi4/tmp/seg/det_val/110.png"
    # if os.path.exists(FILENAME) != 1:
    #     print("The test image does not exist. Exit.")
    #     exit()
    # temp = cv2.imread(FILENAME,0)
    # temp = cv2.resize(temp, (512,512))
    # print(temp.shape)
    # img = temp * int(255 / temp.max())

    # cv2.imwrite("./test_det_result/110.png" , img)


    # Miou(img,img_pred)

    # ###

    # result = steammanager_api.GetProtobuf(STEAMNAME, 0, keyVec)
    # if result.size() == 0:
    #     print("No object detected------------------")
    #     img = cv2.imread(FILENAME)
    #     cv2.imwrite(RESULTFILE, img)
    #     exit()
    # if result[0].errorCode != 0:
    #     print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
    #         result[0].errorCode, result[0].data.decode()))
    #     exit()
    # # process data output from mxpi_objectpostprocessor plugin
    # object_list = MxpiDataType.MxpiObjectList()
    # object_list.ParseFromString(result[0].messageBuf)
    # bounding_boxes = []
    # for obj in object_list.objectVec:
    #     print("class-----------------:", obj)
    #     box = {'x0': int(obj.x0),
    #            'x1': int(obj.x1),
    #            'y0': int(obj.y0),
    #            'y1': int(obj.y1),
    #            'class': int(obj.classVec[0].classId),
    #            'class_name': obj.classVec[0].className,
    #            'confidence': round(obj.classVec[0].confidence, 4)}
    #     bounding_boxes.append(box)
    # img = cv2.imread(FILENAME)
    # # draw each bounding box on the original image
    # for box in bounding_boxes:
    #     class_id = box.get('class')
    #     class_name = box.get('class_name')
    #     score = box.get('confidence')
    #     plot_one_box(img,
    #                  [box.get('x0'),
    #                   box.get('y0'),
    #                   box.get('x1'),
    #                   box.get('y1')],
    #                  cls_id=class_id,
    #                  label=class_name,
    #                  box_score=score)
    # cv2.imwrite(RESULTFILE, img)

    # destroy streams
    steammanager_api.DestroyAllStreams()

