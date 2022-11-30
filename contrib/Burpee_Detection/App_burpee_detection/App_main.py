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

import shutil
import json
import os
import sys
import logging
import cv2
# import time

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client


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

# 正常情况日志级别使用INFO，需要定位时可以修改为DEBUG，此时SDK会打印和服务端的通信信息
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# 1. 设置用户属性, 包括 secret_id, secret_key, region等。App_id 已在CosConfig中移除，请在参数 Bucket 中带上 App_id。Bucket 由 BucketName-App_id 组成
SECRET_ID = 'AKIDq23sVu40iANL5bz93iAPRIxPdleIgjYA'  # 替换为用户的 SecretId，登录https://console.cloud.tencent.com/cam/capi查看
SECRET_KEY = 'QbXIoPlvtd9RUJuHROIxMYVDfsrcrsi2'  # 替换为用户的 SecretKey，登录https://console.cloud.tencent.com/cam/capi查看
REGION = 'ap-shanghai'  # 替换为用户的 region，已创建桶归属的region可在https://console.cloud.tencent.com/cos5/bucket查看
# COS支持的所有region列表参见https://cloud.tencent.com/document/product/436/6224
TOKEN = None  # 如果使用永久密钥不需填入token，若使用临时密钥需填入，临时密钥生成和使用见https://cloud.tencent.com/document/product/436/14048
SCHEME = 'https'  # 指定使用 http/https 协议来访问 COS，默认为 https，可不填

CONFIG = CosConfig(Region=REGION, SecretId=SECRET_ID,
                   SecretKey=SECRET_KEY, Token=TOKEN, Scheme=SCHEME)
CLIENT = CosS3Client(CONFIG)

IMG_NUM = 0
ACTION = ""
ACTION_CNT = 0
STATE = 0
INPUT_COUNT = 0
ERR_FILE = False
FPS = 1
INPUT_PATH = "./input/"
RESULT_PATH = 'result.txt'

# Release the input
if os.path.exists(INPUT_PATH):
    shutil.rmtree(INPUT_PATH)

while True:

    # Check the state of app
    RESPONSE = CLIENT.list_objects(Bucket='burpee-1312708737',
                                   Prefix='state')

    if len(RESPONSE['Contents']) == 2:
        IMG_NUM = 0
        ACTION_CNT = 0
        STATE = 0
        INPUT_COUNT = 0
        if os.path.exists(INPUT_PATH):
            shutil.rmtree(INPUT_PATH)
        continue

    # Check the number of input images
    RESPONSE = CLIENT.list_objects(Bucket='burpee-1312708737',
                                   Prefix='input')

    if len(RESPONSE['Contents']) < IMG_NUM + 2:
        cout << 'wait for inputs' << endl
        continue
    # Check the target input image
    RESPONSE = CLIENT.object_exists(Bucket='burpee-1312708737',
                                    Key='input/img' + str(IMG_NUM) + '.jpg')

    if not RESPONSE:
        cout << 'no such file' << endl
        continue

    # Download the data of input 
    if os.path.exists(INPUT_PATH) != 1:
        os.makedirs("./input/")

    RESPONSE = CLIENT.get_object(Bucket='burpee-1312708737',
                                 Key='input/img' + str(IMG_NUM) + '.jpg')
    RESPONSE['Body'].get_stream_to_file('/input/img' + str(IMG_NUM) + '.jpg')
    cout << 'Get the input successfully' << endl

    # Input object of streams -- detection target   
    IMG_PATH = os.path.join(INPUT_PATH, 'img' + str(IMG_NUM) + '.jpg')

    DATA_INPUT = MxDataInput()
    if os.path.exists(IMG_PATH) != 1:
        cout << 'The image does not exist.' << endl

    with open(IMG_PATH, 'rb') as f:
        DATA_INPUT.data = f.read()

    STREAM_NAME = b'detection'
    IN_PLUGIN_ID = 0
    # Send data to streams by SendDataWithUniqueId()
    UNIQUEID = streamManagerApi.SendDataWithUniqueId(STREAM_NAME, IN_PLUGIN_ID, DATA_INPUT)

    if UNIQUEID < 0:
        cout << 'Failed to send data to stream.' << endl
        exit()

    # Get results from streams by GetResultWithUniqueId()
    INFER_RESULT = streamManagerApi.GetResultWithUniqueId(STREAM_NAME, UNIQUEID, 3000)
    if INFER_RESULT.errorCode != 0:
        cout << 'GetResultWithUniqueId error. errorCode=' << INFER_RESULT.errorCode \
             << ', errorMsg=' << INFER_RESULT.data.decode() << endl
        exit()

    # Get Object class
    RESULTS = json.loads(INFER_RESULT.data.decode())
    IMG = cv2.imread(IMG_PATH)
    IMG_NUM = IMG_NUM + 1

    BEST_CONFIDENCE = 0
    KEY = "MxpiObject"

    if KEY not in RESULTS.keys():
        continue

    # Save the best confidence and its information
    for BBOX in RESULTS['MxpiObject']:
        if round(BBOX['classVec'][0]['confidence'], 4) >= BEST_CONFIDENCE:
            ACTION = BBOX['classVec'][0]['className']
            BEST_CONFIDENCE = round(BBOX['classVec'][0]['confidence'], 4)

    # State change 
    if STATE == 0:
        if ACTION == "crouch":
            STATE = 1
    elif STATE == 1:
        if ACTION == "support":
            STATE = 2
    elif STATE == 2:
        if ACTION == "crouch":
            STATE = 3
    elif STATE == 3:
        if ACTION == "jump":
            STATE = 0
            ACTION_CNT = ACTION_CNT + 1

    # Save txt for results
    FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if os.path.exists(RESULT_PATH):
        os.remove(RESULT_PATH)
    with os.fdopen(os.open('result.txt', FLAGS, 0o755), 'w') as f:
        f.write(str(ACTION_CNT))
    # Upload the result file        
    with open('result.txt', 'rb') as fp:
        RESPONSE = CLIENT.put_object(
            Bucket='burpee-1312708737',
            Body=fp,
            Key='result/result.txt',
            StorageClass='STANDARD',
            EnableMD5=False
        )
    cout << 'upload the result file successfully!!!' << endl

# Destroy All Streams
streamManagerApi.DestroyAllStreams()
