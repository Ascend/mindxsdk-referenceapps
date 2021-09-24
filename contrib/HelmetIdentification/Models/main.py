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

import time
import signal
import cv2
import numpy as np
import StreamManagerApi
import utils
import MxpiDataType_pb2 as MxpiDataType


def my_handler(signum, frame):
    """
    :param signum: signum are used to identify the signal
    :param frame: When the signal occurs, get the status of the process stack
    func:Change flag of stop_stream
    """
    global stop_stream
    stop_stream = True


# exit flag
stop_stream = False
# When about to exit, get the exit signal
signal.signal(signal.SIGINT, my_handler)

# The following belongs to the SDK Process
# init stream manager
streamManagerApi = StreamManagerApi.StreamManagerApi()
ret = streamManagerApi.InitManager()
if ret != 0:
    print("Failed to init Stream manager, ret=%s" % str(ret))

# create streams by pipeline config file
#load  pipline
with open("HelmetDetection.pipline", 'rb') as f:
    pipelineStr = f.read()
ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
# Print error message
if ret != 0:
    print("Failed to create Stream, ret=%s" % str(ret))

# Obtain the inference result by specifying streamName and keyVec
# The data that needs to be obtained is searched by the plug-in name
# Stream name
streamName = b'Detection'
keyVec0 = StreamManagerApi.StringVector()
keyVec0.push_back(b"ReservedFrameInfo")
keyVec0.push_back(b"mxpi_modelinfer0")
keyVec0.push_back(b"mxpi_motsimplesort0")
keyVec0.push_back(b"mxpi_videodecoder0")
keyVec0.push_back(b"mxpi_videodecoder1")

while True:
    # exit flag
    if stop_stream:
        break
    # Get data through GetProtobuf interface
    inferResult0 = streamManagerApi.GetResult(streamName, b'appsink0', keyVec0)
    # Determine whether the output is empty
    if inferResult0.metadataVec.size() == 0:
        print('Object detection result of model infer is null!!!')
        continue

    DictStructure = utils.get_inference_data(inferResult0)
    # the visualization of the inference result, save the output in the specified folder
    utils.cv_visualization(DictStructure[0], DictStructure[1], DictStructure[2], DictStructure[3], DictStructure[4])

# Destroy All Streams
streamManagerApi.DestroyAllStreams()
