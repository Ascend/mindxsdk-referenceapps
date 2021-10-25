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
import io
import os
import cv2
import numpy as np
import sys
import MxpiDataType_pb2 as MxpiDataType
from PIL import Image
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

import faceswap_post

sys.path.append('.')
sys.path.append('../')

FACE1_PATH = sys.argv[1]
FACE2_PATH = sys.argv[2]

STREAM_NAME = b'faceswap'
IN_PLUGIN_ID = 0

POINTS_NUMS = 106  # there are 106 feature points on each face
DATA_NUMS = 212  # 106 points (x ,y): 106*2 = 212

YUV_BYTES_NU = 3
YUV_BYTES_DE = 2

MIN_IMAGE_SIZE = 32
MAX_IMAGE_SIZE = 8192

if __name__ == '__main__':
    # check input image
    input_path = [FACE1_PATH, FACE2_PATH]
    input_image_data = []
    count = 1
    input_valid = False
    for i in input_path:
        # check input image
        if os.path.exists(i) != 1:
            error_message = 'The {} does not exist'.format(i)
            print(error_message)
        else:
            try:
                image = Image.open(i)
                if image.format != 'JPEG':
                    print('input image only support jpg, curr format is.'.format(image.format))
                elif image.width < MIN_IMAGE_SIZE or image.width > MAX_IMAGE_SIZE:
                    print('input image width must in range [32, 8192], curr is {}'.format(image.width))
                elif image.height < MIN_IMAGE_SIZE or image.height > MAX_IMAGE_SIZE:
                    print('input image height must in range [32, 8192], curr is {}'.format(image.height))
                else:
                    input_valid = True
                    # read input image bytes
                    image_bytes = io.BytesIO()
                    image.save(image_bytes, format='JPEG')
                    input_image_data.append(image_bytes.getvalue())
            except IOError:
                print('an IOError occurred while opening {}, maybe your input is not a picture'.format(i))
        if not input_valid:
            print('input image {} is invalid.'.format(i))
            exit(1)
    # initialize the stream manager

    stream_manager = StreamManagerApi()
    stream_state = stream_manager.InitManager()
    if stream_state != 0:
        print("Failed to init Stream manager, stream_state=%s" % str(stream_state))
        exit(1)

    # create streams by pipeline config file
    with open("pipeline/faceswap.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipeline_string = pipeline

    stream_state = stream_manager.CreateMultipleStreams(pipeline_string)
    if stream_state != 0:
        print("Failed to create Stream manager, stream_state=%s" % str(stream_state))
        exit(1)

    # prepare the input of the stream #begin

    # check the face img
    data_input1 = MxDataInput()
    data_input2 = MxDataInput()
    data_input1.data = input_image_data[0]
    data_input2.data = input_image_data[1]
    # prepare the input of the stream #end

    # send the prepared data to the stream
    face_detect_info = []
    landmarks_infer_info = []
    input_face_list = []
    data_input = [data_input1, data_input2]
    for i in data_input:
        # send the prepared data to the stream
        unique_id = stream_manager.SendData(STREAM_NAME, IN_PLUGIN_ID, i)
        if unique_id < 0:
            error_message = 'Failed to send data to stream.'
            print(error_message)
            exit()

        # construct the resulted streamStateurned by the stream
        plugin_names = [b"mxpi_objectpostprocessor0", b"mxpi_tensorinfer1", b"mxpi_imagedecoder0"]
        plugin_vector = StringVector()
        for plugin in plugin_names:
            plugin_vector.push_back(plugin)

        # get the output data according to the relevant plugins
        infer_result = stream_manager.GetProtobuf(STREAM_NAME, IN_PLUGIN_ID, plugin_vector)

        # checking whether the infer results is valid or not
        if infer_result.size() == 0:
            error_message = 'inferResult is null, please check the stream log for details'
            print(error_message)
            exit()

        if infer_result[0].errorCode != 0:
            error_message = 'Unable to get effective infer results, please check the stream log for details'
            print(error_message)
            exit()

        # the output information of "mxpi_objectpostprocessor0"
        object_list = MxpiDataType.MxpiObjectList()
        object_list.ParseFromString(infer_result[0].messageBuf)

        # only select the image with a "face" label
        for item in object_list.objectVec:
            try:
                if item.classVec[0].className == "face":
                    face_detect_info.append(object_list.objectVec[0])
                else:
                    error_message = "The model cannot detect the obvious face in this picture, " \
                                    "please input another image"
                    print(error_message)
                    exit()
            except IndexError:
                error_message = "The yolov4 model cannot detect anything in this picture," \
                                " please change another image."
                print(error_message)
                exit()

        # the output information of "mxpi-tensorinfer1" which is used to detect the features points of a crop face
        points_infer_list = MxpiDataType.MxpiTensorPackageList()
        points_infer_list.ParseFromString(infer_result[1].messageBuf)
        res = np.frombuffer(points_infer_list.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        landmarks_infer_info.append(res.tolist())

        # to get the input image information
        decode_list = MxpiDataType.MxpiVisionList()
        decode_list.ParseFromString(infer_result[2].messageBuf)
        decode_data = decode_list.visionVec[0].visionData.dataStr
        decode_info = decode_list.visionVec[0].visionInfo

        img_yuv = np.frombuffer(decode_data, np.uint8)
        img_bgr = img_yuv.reshape(decode_info.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, decode_info.widthAligned)
        img = cv2.cvtColor(img_bgr, getattr(cv2, "COLOR_YUV2BGR_NV12"))
        source_face = img[0:decode_info.height, 0:decode_info.width]
        input_face_list.append(source_face)

    # swap the cropped face # begin
    # get the landmark points of faces
    face1_points = landmarks_infer_info[0]
    face2_points = landmarks_infer_info[1]

    # the rectangular coordinate points of face detection #begin
    crop_face1_left = face_detect_info[0].x0
    crop_face1_right = face_detect_info[0].x1
    crop_face1_bottom = face_detect_info[0].y0
    crop_face1_top = face_detect_info[0].y1
    crop_face1_width = crop_face1_right - crop_face1_left
    crop_face1_height = crop_face1_top - crop_face1_bottom

    crop_face2_left = face_detect_info[1].x0
    crop_face2_right = face_detect_info[1].x1
    crop_face2_bottom = face_detect_info[1].y0
    crop_face2_top = face_detect_info[1].y1
    crop_face2_width = crop_face2_right - crop_face2_left
    crop_face2_height = crop_face2_top - crop_face2_bottom
    # the rectangular coordinate points of face detection #end

    #  get the cropped face #begin
    base_face = input_face_list[0]
    cover_face = input_face_list[1]
    crop_base_face = base_face[int(crop_face1_bottom): int(crop_face1_top),
                     int(crop_face1_left): int(crop_face1_right)]
    crop_cover_face = cover_face[int(crop_face2_bottom): int(crop_face2_top),
                      int(crop_face2_left):int(crop_face2_right)]
    #  get the cropped face #end

    for i in range(0, DATA_NUMS):
        if i % 2 == 0:
            face1_points[i] = int(face1_points[i] * crop_face1_width)
            face2_points[i] = int(face2_points[i] * crop_face2_width)
        else:
            face1_points[i] = int(face1_points[i] * crop_face1_height)
            face2_points[i] = int(face2_points[i] * crop_face2_height)
    base_points = np.array(face1_points, dtype=np.int32).reshape(POINTS_NUMS, 2)
    cover_points = np.array(face2_points, dtype=np.int32).reshape(POINTS_NUMS, 2)
    baseLandmarks = np.mat(base_points)
    coverLandmarks = np.mat(cover_points)

    # scan <faceswap_post.py> for more details of this process
    faceswap_post.swap_face(baseLandmarks, coverLandmarks, crop_base_face, crop_cover_face)

    # swap the cropped face #end
    face_swap_result = cv2.imread("./only_face_swap.jpg")

    # merge the face_swap result into the source image
    base_face[int(crop_face1_bottom): int(crop_face1_top),
    int(crop_face1_left):int(crop_face1_right)] = face_swap_result
    cv2.imwrite("face_swap_result.jpg", base_face)

    # delete the intermediate process picture
    os.remove("./only_face_swap.jpg")

    # destroy streams
    stream_manager.DestroyAllStreams()
