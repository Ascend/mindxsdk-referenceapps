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

import signal
import cv2
import numpy as np
import StreamManagerApi
import utils


def my_handler(signum, frame):
    """
    :param signum: signum are used to identify the signal
    :param frame: When the signal occurs, get the status of the process stack
    func:Change flag of STOP_STREAM
    """
    global STOP_STREAM
    STOP_STREAM = True


# exit flag
STOP_STREAM = False
# When about to exit, get the exit signal
signal.signal(signal.SIGINT, my_handler)

# The following belongs to the SDK Process
# init stream manager
STREAMMANAGERAPI = StreamManagerApi.StreamManagerApi()
RET = STREAMMANAGERAPI.InitManager()
if RET != 0:
    print("Failed to init Stream manager, ret=%s" % str(RET))

# create streams by pipeline config file
#load  pipline
with open("HelmetDetection.pipline", 'rb') as f:
    PIPELINESTR = f.read()
RET = STREAMMANAGERAPI.CreateMultipleStreams(PIPELINESTR)
# Print error message
if RET != 0:
    print("Failed to create Stream, ret=%s" % str(RET))

# Obtain the inference result by specifying streamName and keyVec
# The data that needs to be obtained is searched by the plug-in name
# Stream name
STREAMNAME = b'Detection'
KEYVEC0 = STREAMMANAGERAPI.StringVector()
KEYVEC0.push_back(b"ReservedFrameInfo")
KEYVEC0.push_back(b"mxpi_modelinfer0")
KEYVEC0.push_back(b"mxpi_motsimplesort0")
KEYVEC0.push_back(b"mxpi_videodecoder0")
KEYVEC0.push_back(b"mxpi_videodecoder1")

while True:
    # exit flag
    if STOP_STREAM:
        break
    # Get data through GetProtobuf interface
    INFERRESULT0 = STREAMMANAGERAPI.GetProtobuf(STREAMNAME, 0, KEYVEC0)
    # output errorCode
    if INFERRESULT0[0].errorCode != 0:
        # Print error message
        if INFERRESULT0[0].errorCode == 1001:
            print('Object detection result of model infer is null!!!')
        continue

    # Take the inference result from the corresponding data structure
    DICTSTRUCTURE = utils.get_inference_data(INFERRESULT0)
    # the visualization of the inference result, save the output in the specified folder
    utils.cv_visualization(DICTSTRUCTURE[0], DICTSTRUCTURE[1], DICTSTRUCTURE[2], DICTSTRUCTURE[3], DICTSTRUCTURE[4])

# Destroy All Streams
STREAMMANAGERAPI.DestroyAllStreams()
