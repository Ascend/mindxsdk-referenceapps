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
import time
import cv2

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


class OStream:
    def __init__(self, file):
        self.file = file

    def __lshift__(self, obj):
        self.file.write(str(obj))
        return self


cout = OStream(sys.stdout)
END_L = '/n'

# The following belongs to the SDK Process
streamManagerApi = StreamManagerApi()
# Init stream manager
ret = streamManagerApi.InitManager()
if ret != 0:
    cout << 'Failed to init Stream manager, ret=' << str(ret) << END_L
    exit()
# Mark start time
start = time.time()
# Create streams by pipeline config file
# Load  pipline
with open("../pipeline/burpee_detection_p.pipeline", 'rb') as f:
    PIPELINE_STR = f.read()
ret = streamManagerApi.CreateMultipleStreams(PIPELINE_STR)
# Print error message
if ret != 0:
    cout << 'Failed to create Stream, ret=' << str(ret) << END_L
    exit()

DET_IMG_COUNT = 0  # the number of detected pictures

# Init the directory of input and output
INPUT_PATH = ["../data/images/test/"]  # the path of input

OUTPUT_PATH = ["./result_test/"]  # the output path of txt file

OUTPUT_PIC_PATH = ["./result_test_pic/"]  # the output path of pictures

for index, path in enumerate(INPUT_PATH):

    RESULT_PATH = OUTPUT_PATH[index]

    # Create the output directory
    if os.path.exists(RESULT_PATH) != 1:
        os.makedirs(RESULT_PATH)
    else:
        shutil.rmtree(RESULT_PATH)
        os.makedirs(RESULT_PATH)

    if os.path.exists(OUTPUT_PIC_PATH[index]) != 1:
        os.makedirs(OUTPUT_PIC_PATH[index])
    else:
        shutil.rmtree(OUTPUT_PIC_PATH[index])
        os.makedirs(OUTPUT_PIC_PATH[index])

    # Input object of streams -- detection target
    for item in os.listdir(path):
        IMG_PATH = os.path.join(path, item)
        cout << 'read file path:' << IMG_PATH << END_L
        IMG_NAME = os.path.splitext(item)[0]
        IMG_TXT = RESULT_PATH + IMG_NAME + ".txt"
        if os.path.exists(IMG_TXT):
            os.remove(IMG_TXT)
        DATA_INPUT = MxDataInput()
        if os.path.exists(IMG_PATH) != 1:
            cout << 'The image does not exist.' << END_L
            continue
        with open(IMG_PATH, 'rb') as f:
            DATA_INPUT.data = f.read()
        STREAM_NAME = b'detection'
        IN_PLUGIN_ID = 0
        # Send data to streams by SendDataWithUniqueId()
        UNIQUE_ID = streamManagerApi.SendDataWithUniqueId(STREAM_NAME, IN_PLUGIN_ID, DATA_INPUT)

        if UNIQUE_ID < 0:
            cout << 'Failed to send data to stream.' << END_L
            exit()

        # Get results from streams by GetResultWithUniqueId()
        INFER_RESULT = streamManagerApi.GetResultWithUniqueId(STREAM_NAME, UNIQUE_ID, 3000)
        if INFER_RESULT.errorCode != 0:
            cout << 'GetResultWithUniqueId error. errorCode=' << INFER_RESULT.errorCode \
            << ', errorMsg=' << INFER_RESULT.data.decode() << END_L
            exit()

        DET_IMG_COUNT = DET_IMG_COUNT + 1

        # Get ObjectList
        RESULTS = json.loads(INFER_RESULT.data.decode())

        IMG = cv2.imread(IMG_PATH)
        BBOXES = []
        best_class = {}
        TEXT = ""
        BEST_CONFIDENCE = 0
        KEY = "MxpiObject"
        if KEY not in RESULTS.keys():
            continue
        for BBOX in RESULTS['MxpiObject']:
            BBOXES = {'x0': int(BBOX['x0']),
                      'x1': int(BBOX['x1']),
                      'y0': int(BBOX['y0']),
                      'y1': int(BBOX['y1']),
                      'confidence': round(BBOX['classVec'][0]['confidence'], 4),
                      'text': BBOX['classVec'][0]['className']}
            key_value = BBOXES.get('confidence', "abc")
            if key_value:
                pass
            else:
                continue
            if key_value > BEST_CONFIDENCE:
                L1 = []
                # Convert the label as Yolo label
                x_center = round((BBOXES['x1'] + BBOXES['x0']) * 0.5 / IMG.shape[1], 6)
                y_center = round((BBOXES['y1'] + BBOXES['y0']) * 0.5 / IMG.shape[0], 6)
                w_nor = round((BBOXES['x1'] - BBOXES['x0']) / IMG.shape[1], 6)
                h_nor = round((BBOXES['y1'] - BBOXES['y0']) / IMG.shape[0], 6)
                L1.append(x_center)
                L1.append(y_center)
                L1.append(w_nor)
                L1.append(h_nor)
                L1.append(BBOXES['confidence'])
                L1.append(BBOXES['text'])
                BEST_CONFIDENCE = BBOXES['confidence']
                TEXT = "{}{}".format(str(BBOXES['confidence']), " ")
                for CONTENT in BBOXES['text']:
                    TEXT += CONTENT
                best_class = {'x0': int(BBOX['x0']),
                              'x1': int(BBOX['x1']),
                              'y0': int(BBOX['y0']),
                              'y1': int(BBOX['y1']),
                              'confidence': round(BBOX['classVec'][0]['confidence'], 4),
                              'text': BBOX['classVec'][0]['className']}
        # Draw rectangle and txt for visualization
        key_value = (best_class.get('x0', "abc") and best_class.get('y0', "abc")) and \
                    (best_class.get('x1', "abc") and best_class.get('y1', "abc"))
        if key_value:
            pass
        else:
            continue
        cv2.putText(IMG, TEXT, (best_class['x0'] + 10, best_class['y0'] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 100, 255), 2)
        cv2.rectangle(IMG, (best_class['x0'], best_class['y0']), (best_class['x1'], best_class['y1']),
                      (255, 0, 0), 2)

        # Save picture
        originImgFile = OUTPUT_PIC_PATH[index] + IMG_NAME + '.jpg'
        cv2.imwrite(originImgFile, IMG)

        # Save txt for results
        FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        with os.fdopen(os.open(IMG_TXT, FLAGS, 0o755), 'w') as f:
            CONTENT = '{} {} {} {} {} {}'.format(L1[5], L1[4], L1[0], L1[1], L1[2], L1[3])
            f.write(CONTENT)
            f.write('\n')

end = time.time()
cost_time = end - start
# Mark spend time
cout << 'Image count:' << DET_IMG_COUNT << END_L
cout << 'Spend time:' << cost_time << END_L
cout << 'fps:' << (DET_IMG_COUNT / cost_time) << END_L
# Destroy All Streams
streamManagerApi.DestroyAllStreams()
