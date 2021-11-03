#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from PIL import Image
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import matplotlib.pyplot as plt
import matplotlib as mpl
from util.visual_utils import semantic_to_mask, final_result_create, enable_contrast_output, decode_seg_map

# Drawing font Settings and Display Chinese label
plt.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def softmax(x):
    """
        Compute softmax values for each sets of scores in x.
        Args:
            matrix x
        Returns:
            softmax values for x
    """
    mx = np.max(x, axis=1, keepdims=True)
    numerator = np.exp(x - mx)
    denominator = np.sum(numerator, axis=1, keepdims=True)
    return numerator / denominator


def run_pipeline(arg_arr):
    """
        enable comparison graph output
        Args:
            arg_arr: arg_arr[0] is the tested image, arg_arr[1] is whether to enable comparison graph output, arg_arr[2]
                     is the output directory of the comparison graph result
        Returns:
            null
        Output:
            a comparison graph result in arr[2]
    """
    # The tested image setting
    img_dir = arg_arr[0]
    # Whether to enable comparison graph output
    contrast_output = arg_arr[1]
    # Output directory of the comparison graph result
    contrast_output_dir = arg_arr[2]
    streamManagerApi = StreamManagerApi()

    # Init stream manager
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # Create streams by pipeline pipeline config file
    with open("pipeline/segmentation.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipelineStr = pipeline
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    dataInput = MxDataInput()
    if os.path.exists(img_dir) != 1:
        print("The test image " + str(img_dir) + " does not exist.")
        exit()

    # Judge the input picture whether is a jpg format
    try:
        image = Image.open(img_dir)
        if image.format != 'JPEG':
            print('The input image is not the jpg format.')
            exit()
        elif image.height != 256 or image.width != 256:
            print('The input image width must be 256*256, curr is {}*{}'.
                  format(image.width, image.height))
            exit()
    except IOError:
        print('an IOError occurred while opening {}, maybe your input is not a picture'.format(img_dir))
        exit()

    input_format = img_dir.split('.')[1]
    if input_format != 'jpg':
        print("The input image file is not the jpg suffix.")
        exit()

    with open(img_dir, 'rb') as f:
        dataInput.data = f.read()

    # Inputs models to a specified stream based on streamName.
    streamName = b'segmentation'
    inPluginId = 0

    uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)

    if uniqueId < 0:
        print("Failed to send models to stream.")
        exit()

    # Get result numpy array
    keys = [b"mxpi_tensorinfer0", b"mxpi_tensorinfer1"]
    key_vec = StringVector()
    for key in keys:
        key_vec.push_back(key)

    # Get inference result
    infer = streamManagerApi.GetResult(streamName, b'appsink0', key_vec)
    if infer.metadataVec[0].errorCode != 0:
        print("GetResult error. errorCode=%d ,errorMsg=%s" % (
            infer.metadataVec[0].errorCode, infer.metadataVec[0].errorMsg))
        exit()

    tensorList = MxpiDataType.MxpiTensorPackageList()

    # Get the result of the DANet model
    tensorList.ParseFromString(infer.metadataVec[0].serializedMetadata)
    output_res_DANet = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
    # reshape the matrix to (1, 8, 256, 256)
    tensor_res_DANet = output_res_DANet.reshape(1, 8, 256, 256)
    pred_DANet = softmax(tensor_res_DANet)

    # Get the result of the Deeplab_v3 model
    tensorList.ParseFromString(infer.metadataVec[1].serializedMetadata)
    output_res_Deeplab = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
    # reshape the matrix to (1, 8, 256, 256)
    tensor_res_Deeplab = output_res_Deeplab.reshape(1, 8, 256, 256)
    pred_Deeplab = softmax(tensor_res_Deeplab)

    pred_final = pred_Deeplab + pred_DANet

    # 8 kinds of semantic tags
    labels = [100, 200, 300, 400, 500, 600, 700, 800]
    # Colors corresponding to 8 semantic tags
    label_colours = np.array(
        [[255, 127, 39], [255, 202, 24], [253, 236, 166], [236, 28, 36], [184, 61, 186],
         [0, 168, 243], [88, 88, 88], [14, 209, 69]])
    # Semantic diagram to label diagram
    pred_final = semantic_to_mask(pred_final, labels).squeeze().astype(np.uint16)
    # The result is mapped to a picture
    rgb_res_pic = decode_seg_map(pred_final, labels, label_colours)
    # Result image post-processing
    final_result_create(rgb_res_pic, label_colours)
    # Enable comparison graph output if true
    if contrast_output == 'True':
        enable_contrast_output([img_dir, 'result/temp_result/result.jpg', contrast_output_dir])
    print('success!!!!!!!!!!!!!')

    # Destroy streams
    streamManagerApi.DestroyAllStreams()


if __name__ == '__main__':
    arg_arr_input_output_path = []
    if len(sys.argv) != 4:
        print('Wrong parameter setting.')
        exit()
    if sys.argv[2] != 'True' and sys.argv[2] != 'False':
        print('The second parameter must be True or False.')
        exit()

    for i in range(1, len(sys.argv)):
        arg_arr_input_output_path.append(sys.argv[i])
    run_pipeline(arg_arr_input_output_path)
