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
# Load  pipline
with open("../pipeline/burpee_detection_p.pipeline", 'rb') as f:
    pipelineStr = f.read()
ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
# Print error message
if ret != 0:
    print("Failed to create Stream, ret=%s" % str(ret))
    exit()

detImg_count = 0      #the number of detected pictures
# Init the directory of input and output
PATH = ["../data/images/dark/","../data/images/multi/","../data/images/png/","../data/images/empty/","../data/images/forward/","../data/images/test/"]    #the path of input

Result_PATH = ["./result_dark/","./result_multi/","./result_png/","./result_empty/","./result_forward/","./result_test/"]                                 #the output path of txt file

Result_Pic_PATH = ["./result_dark_pic/","./result_multi_pic/","./result_png_pic/","./result_empty_pic/","./result_forward_pic/","./result_test_pic/"]     #the output path of pictures

for index,path in enumerate(PATH):
  
  result_path = Result_PATH[index]
  
  # Create the output directory
  if os.path.exists(result_path) != 1:
    os.makedirs(result_path)
  else:
    shutil.rmtree(result_path)
    os.makedirs(result_path)
    
  if os.path.exists(Result_Pic_PATH[index]) != 1:
    os.makedirs(Result_Pic_PATH[index])
  else:
    shutil.rmtree(Result_Pic_PATH[index])
    os.makedirs(Result_Pic_PATH[index])
  
  # Input object of streams -- detection target
  for item in os.listdir(path):
      img_path = os.path.join(path,item)
      print("read file path:",img_path)
      img_name = os.path.splitext(item)[0]
      img_txt = result_path + img_name + ".txt"
      if os.path.exists(img_txt):
          os.remove(img_txt)
      dataInput = MxDataInput()
      if os.path.exists(img_path) != 1:
          print("The image does not exist.")
          continue
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
      
      detImg_count = detImg_count + 1
      
      # Get ObjectList
      results = json.loads(infer_result.data.decode())
      
      img = cv2.imread(img_path)
      bboxes = []
      best_confidence = 0
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

          if bboxes['confidence'] > best_confidence:
            L1 = []
            # Convert the label as Yolo label
            x_center = round((bboxes['x1']+bboxes['x0'])*0.5/img.shape[1],6)
            y_center = round((bboxes['y1']+bboxes['y0'])*0.5/img.shape[0],6)
            w_nor = round((bboxes['x1']-bboxes['x0'])/img.shape[1],6)
            h_nor = round((bboxes['y1']-bboxes['y0'])/img.shape[0],6)
            L1.append(x_center)
            L1.append(y_center)
            L1.append(w_nor)
            L1.append(h_nor)
            L1.append(bboxes['confidence'])
            L1.append(bboxes['text'])
            best_confidence = bboxes['confidence']
            text = "{}{}".format(str(bboxes['confidence']), " ")
            for item in bboxes['text']:
              text += item
            best_class = {'x0': int(bbox['x0']),
                          'x1': int(bbox['x1']),
                          'y0': int(bbox['y0']),
                          'y1': int(bbox['y1']),
                          'confidence': round(bbox['classVec'][0]['confidence'], 4),
                          'text': bbox['classVec'][0]['className']}
      # Draw rectangle and txt for visualization
      cv2.putText(img, text, (best_class['x0'] + 10, best_class['y0'] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)
      cv2.rectangle(img, (best_class['x0'], best_class['y0']), (best_class['x1'], best_class['y1']), (255, 0, 0), 2)
    
      # Save picture
      oringe_imgfile = Result_Pic_PATH[index] + img_name + '.jpg'
      cv2.imwrite(oringe_imgfile, img)
  
      # Save txt for results
      with open(img_txt,"a+") as f:
          content = '{} {} {} {} {} {}'.format(L1[5], L1[4], L1[0], L1[1], L1[2], L1[3])
          f.write(content)
          f.write('\n')

      
      
end = time.time()
cost_time = end - start
# Mark spend time
print("Image count:%d" % detImg_count)
print("Spend time:%10.3f" % cost_time)
print("fps:%10.3f" % (detImg_count/cost_time))
# Destroy All Streams
streamManagerApi.DestroyAllStreams()
