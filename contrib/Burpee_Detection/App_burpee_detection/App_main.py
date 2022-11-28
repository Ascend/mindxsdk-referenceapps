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

import shutil
import json
import os
import time
import cv2
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import sys
import logging


# The following belongs to the SDK Process
streamManagerApi = StreamManagerApi()
# Init stream manager
ret = streamManagerApi.InitManager()
if ret != 0:
    print("Failed to init Stream manager, ret=%s" % str(ret))
    exit()
# Mark start time
start = time.time()
# Create streams by pipeline config file
# load  pipline
with open("../pipeline/burpee_detection_p.pipeline", 'rb') as f:
    pipelineStr = f.read()
ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
# Print error message
if ret != 0:
    print("Failed to create Stream, ret=%s" % str(ret))
    exit()


# 正常情况日志级别使用INFO，需要定位时可以修改为DEBUG，此时SDK会打印和服务端的通信信息
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# 1. 设置用户属性, 包括 secret_id, secret_key, region等。Appid 已在CosConfig中移除，请在参数 Bucket 中带上 Appid。Bucket 由 BucketName-Appid 组成
secret_id = 'AKIDq23sVu40iANL5bz93iAPRIxPdleIgjYA'     # 替换为用户的 SecretId，请登录访问管理控制台进行查看和管理，https://console.cloud.tencent.com/cam/capi
secret_key = 'QbXIoPlvtd9RUJuHROIxMYVDfsrcrsi2'   # 替换为用户的 SecretKey，请登录访问管理控制台进行查看和管理，https://console.cloud.tencent.com/cam/capi
region = 'ap-shanghai'      # 替换为用户的 region，已创建桶归属的region可以在控制台查看，https://console.cloud.tencent.com/cos5/bucket
                           # COS支持的所有region列表参见https://cloud.tencent.com/document/product/436/6224
token = None               # 如果使用永久密钥不需要填入token，如果使用临时密钥需要填入，临时密钥生成和使用指引参见https://cloud.tencent.com/document/product/436/14048
scheme = 'https'           # 指定使用 http/https 协议来访问 COS，默认为 https，可不填

config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token, Scheme=scheme)
client = CosS3Client(config)

img_num = 0
action_cnt = 0
state = 0
input_count = 0
err_file = False
fps = 1
input_path = "./input/"
result_path = 'result.txt'

# Release the input
if os.path.exists(input_path):
    shutil.rmtree(input_path)


while True:
    
    # Check the state of app
    response = client.list_objects(
    Bucket='burpee-1312708737',
    Prefix='state')

    if len(response['Contents']) == 2:
        img_num = 0
        action_cnt = 0
        state = 0
        input_count = 0
        if os.path.exists(input_path):
            shutil.rmtree(input_path)
        continue

    # Check the number of input images
    response = client.list_objects(
    Bucket='burpee-1312708737',
    Prefix='input')

    if len(response['Contents']) < img_num + 2:
        print("wait for inputs")
        continue
    # Check the target input image
    response = client.object_exists(
        Bucket='burpee-1312708737',
        Key='input/img'+ str(img_num) +'.jpg')

    if not response:
          print("no such file")
          continue

    # Download the data of input 
    if os.path.exists(input_path) != 1:
        os.makedirs("./input/")
        
    response = client.get_object(
              Bucket='burpee-1312708737',
              Key= 'input/img'+ str(img_num) +'.jpg'
              )
    response['Body'].get_stream_to_file('/home/HwHiAiUser/Burpee_Detection/Burpee_Detection/App_burpee_detection/input/img'+ str(img_num) +'.jpg')
    print("Get the input successfully")
    
    # Input object of streams -- detection target   
    img_path = os.path.join(input_path,'img'+str(img_num)+'.jpg')
    
    dataInput = MxDataInput()
    if os.path.exists(img_path) != 1:
        print("The image does not exist.")

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
    
    # Get Objectclass
    results = json.loads(infer_result.data.decode())
    img = cv2.imread(img_path)
    img_num = img_num + 1

    best_confidence = 0
    key = "MxpiObject"
    
    if key not in results.keys():   
        continue
        
    # Save the best confidence and its information
    for bbox in results['MxpiObject']:
        if round(bbox['classVec'][0]['confidence'], 4) >= best_confidence:           
            action = bbox['classVec'][0]['className']
            best_confidence = round(bbox['classVec'][0]['confidence'], 4)


    # State change 
    if state == 0:
        if action == "crouch":
            state = 1
    elif state == 1:
        if action == "support":
            state = 2
    elif state == 2:
        if action == "crouch":
            state = 3
    elif state == 3:
        if action == "jump":
            state = 0
            action_cnt = action_cnt + 1
          
    # Save txt for results
    if os.path.exists(result_path):
        os.remove(result_path)
    with open('result.txt',"a+") as f:
        f.write(str(action_cnt))   
    # Upload the result file        
    with open('result.txt', 'rb') as fp:
        response = client.put_object(
            Bucket='burpee-1312708737',
            Body=fp,
            Key='result/result.txt',
            StorageClass='STANDARD',
            EnableMD5=False
            )   
    print("upload the result file successfully!!!")

# Destroy All Streams
streamManagerApi.DestroyAllStreams()
