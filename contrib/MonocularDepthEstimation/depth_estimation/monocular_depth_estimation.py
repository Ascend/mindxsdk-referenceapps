#!/usr/bin/env python
# -*- encoding: utf-8 -*-
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

import MxpiDataType_pb2 as MxpiDataType
import numpy as np

from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


def infer(stream_manager, stream_name, in_plugin_id, data_input):
    """
    send data into infer stream and get infer result
    :param stream_manager: the manager of infer streams
    :param stream_name: name of infer stream that needs to operate
    :param in_plugin_id: ID of the plug-in that needs to send data
    :param data_input: data that needs to send into infer stream
    :return: infer results
    """
    # Inputs data to a specified stream based on streamName.
    unique_id = stream_manager.SendData(stream_name, in_plugin_id, data_input)
    if unique_id < 0:
        error_message = 'Failed to send data to stream'
        raise IOError(error_message)

    # construct output plugin vector
    plugin_names = [b"mxpi_tensorinfer0", b"mxpi_imagedecoder0"]
    plugin_vec = StringVector()
    for key in plugin_names:
        plugin_vec.push_back(key)

    # get plugin output data
    infer_result = stream_manager.GetProtobuf(stream_name, in_plugin_id, plugin_vec)

    # check whether the inferred results is valid
    if infer_result.size() == 0:
        error_message = 'unable to get effective infer results, please check the stream log for details'
        raise IOError(error_message)
    if infer_result[0].errorCode != 0:
        error_message = "GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            infer_result[0].errorCode, infer_result[0].data.decode())
        raise AssertionError(error_message)

    # get mxpi_tensorinfer0 output data
    infer_result_list = MxpiDataType.MxpiTensorPackageList()
    infer_result_list.ParseFromString(infer_result[0].messageBuf)

    # get mxpi_imagedecoder0 output data
    vision_list = MxpiDataType.MxpiVisionList()
    vision_list.ParseFromString(infer_result[1].messageBuf)

    # get input picture size
    input_pic_info = vision_list.visionVec[0].visionInfo
    input_pic_height = input_pic_info.heightAligned
    input_pic_width = input_pic_info.widthAligned

    # get output depth pic size
    output_depth_pic_height = infer_result_list.tensorPackageVec[0].tensorVec[0].tensorShape[2]
    output_depth_pic_width = infer_result_list.tensorPackageVec[0].tensorVec[0].tensorShape[3]

    # get output depth pic data
    output_depth_info_data = infer_result_list.tensorPackageVec[0].tensorVec[0].dataStr
    # converting the byte data into little-endian 32 bit float array ('<f4')
    depth_info = np.frombuffer(output_depth_info_data, dtype='<f4')
    depth_info = depth_info.reshape(output_depth_pic_height, output_depth_pic_width)

    return input_pic_width, input_pic_height, depth_info


def depth_estimation(images_data, is_batch=False):
    """
    get depth info of input images
    :param images_data: binary data of input images
    :param is_batch: whether is batch mode (True represents that need to process plural pictures)
    :return: depth info and image info of input data
    """
    stream_manager = StreamManagerApi()
    error_message = 'program common error'

    # init stream manager
    ret = stream_manager.InitManager()
    if ret != 0:
        error_message = "Failed to init Stream manager, ret=%s" % str(ret)
        raise ConnectionError(error_message)

    # create streams by pipeline config file
    with open('pipeline/depth_estimation.pipeline', 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipeline_str = pipeline
    ret = stream_manager.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        error_message = "Failed to create Stream, ret=%s" % str(ret)
        raise IOError(error_message)

    # config
    stream_name = b'estimation'
    in_plugin_id = 0

    # infer results
    input_images_info = np.empty(shape=(0, 2))
    images_depth_info = np.empty(shape=(0, 240, 320))

    # Construct the input of the stream
    data_input = MxDataInput()
    if not is_batch:
        data_input.data = images_data
        # model infer
        input_pic_width, input_pic_height, depth_info = infer(stream_manager, stream_name, in_plugin_id, data_input)
        # save data
        input_images_info = np.vstack([input_images_info, [input_pic_height, input_pic_width]])
        images_depth_info = np.vstack([images_depth_info, [depth_info]])
        print('processed image: height = {} width = {}'.format(input_pic_height, input_pic_width))
    else:
        index = 0
        for image in images_data:
            data_input.data = image

            # model infer
            input_pic_width, input_pic_height, depth_info = infer(stream_manager, stream_name, in_plugin_id, data_input)

            # save infer result
            index += 1
            input_images_info = np.vstack([input_images_info, [input_pic_height, input_pic_width]])
            images_depth_info = np.vstack([images_depth_info, [depth_info]])
            print('processed {}-th image: height = {} width = {}'.format(index, input_pic_height, input_pic_width))

    # destroy streams
    stream_manager.DestroyAllStreams()

    return images_depth_info, input_images_info
