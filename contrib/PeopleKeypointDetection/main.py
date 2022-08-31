#!/usr/bin/env python
# coding=utf-8

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

import json
import os
import math
import configparser
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxProtobufIn, MxDataInput, StringVector, InProtobufVector

NAME_MAX_LEN = 100;
IMG_PATH = 'test.jpg'
BBOX_REAL = [2000, 2000]
SKELETON = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11),
             (8, 9), (9, 10), (11, 12), (12, 13), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7) )


def read_config(config_fname):
    curpath = os.path.dirname(os.path.realpath(__file__))
    cfgpath = os.path.join(curpath, config_fname)

    conf = configparser.ConfigParser()
    conf.read(cfgpath, encoding = "utf-8")

    return conf


def get_config_data(net_param):
    items_intrinsic = net_param.items("intrinsic")
    for i in range(len(items_intrinsic)):
        boxes_str = net_param.get("intrinsic", items_intrinsic[i][0])
        intrinsic_m = json.loads(boxes_str)

    keypoints_cam_n = []
    items_keypoints_cam = net_param.items("keypoints_cam")
    for i in range(len(items_keypoints_cam)):
        boxes_str = net_param.get("keypoints_cam", items_keypoints_cam[i][0])
        keypoints_cam_n.append(json.loads(boxes_str))

    keypoints_img_n = []
    items_keypoints_img = net_param.items("keypoints_img")
    for i in range(len(items_keypoints_img)):
        boxes_str = net_param.get("keypoints_img", items_keypoints_img[i][0])
        keypoints_img_n.append(json.loads(boxes_str))

    if len(keypoints_cam_n) != len(keypoints_img_n):
        print("keypoints_cam_n and keypoints_img_n is error")
        exit()

    return keypoints_cam_n, keypoints_img_n, intrinsic_m


def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)
    kps_len = len(kps_lines)

    # Draw the keypoints.
    for l in range(kps_len):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        person_num = np.array(kps).shape[0]
        kps = np.array(kps)
        
        for n in range(person_num):
            p1 = kps[n][i1][0].astype(np.int32), kps[n][i1][1].astype(np.int32)
            p2 = kps[n][i2][0].astype(np.int32), kps[n][i2][1].astype(np.int32)
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_3d_multiple_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]
    kps_len = len(kps_lines)

    for l in range(kps_len):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        person_num = np.array(kpt_3d).shape[0]

        for n in range(person_num):
            x = np.array([kpt_3d[n][i1][0], kpt_3d[n][i2][0]])
            y = np.array([kpt_3d[n][i1][1], kpt_3d[n][i2][1]])
            z = np.array([kpt_3d[n][i1][2], kpt_3d[n][i2][2]])

            if kpt_3d_vis[n][i1][0] > 0 and kpt_3d_vis[n][i2][0] > 0:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d_vis[n][i1][0] > 0:
                ax.scatter(kpt_3d[n][i1][0], kpt_3d[n][i1][2], -kpt_3d[n][i1][1], c=colors[l], marker='o')
            if kpt_3d_vis[n][i2][0] > 0:
                ax.scatter(kpt_3d[n][i2][0], kpt_3d[n][i2][2], -kpt_3d[n][i2][1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()
    plt.savefig("output_root_3d_pose.png")


def get_k_value(bbox_n, intrinsic_nn):
    print(type(BBOX_REAL[0]))
    print(type(intrinsic_nn[0]))
    print(type(bbox_n[2]))
    k_value_n = np.array([math.sqrt(BBOX_REAL[0] * BBOX_REAL[1] * intrinsic_nn[0] * intrinsic_nn[1] /
                                    (bbox_n[2] * bbox_n[3]))]).astype(np.float32)
    return k_value_n


def process_bbox(bbox, width, height):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    input_shape = [256, 256]
    aspect_ratio = input_shape[1]/input_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    return bbox


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return cam_coord


def evaljosn(rebbox, root3d, intrinsic_nn):
    print('make Evaluation json start...')
    pred_save = []
    sample_num = len(rebbox)
    fx = intrinsic_nn[0]
    fy = intrinsic_nn[1]
    cx = intrinsic_nn[2]
    cy = intrinsic_nn[3]
    f = np.array([fx, fy])
    c = np.array([cx, cy])
    output_shape = [256//4, 256//4]
    for n in range(sample_num):
        image_id = 0
        bbox = rebbox[n]
        score = 1
        # restore coordinates to original space
        pred_root = root3d[n].copy()
        pred_root[0] = pred_root[0] / output_shape[1] * bbox[2] + bbox[0]
        pred_root[1] = pred_root[1] / output_shape[0] * bbox[3] + bbox[1]

        # back project to camera coordinate system
        pred_root = pixel2cam(pred_root[None, :], f, c)[0]

        pred_save.append({'image_id': image_id, 'root_cam': pred_root.tolist(), 'bbox': bbox.tolist(), 'score': score})

    output_path = 'bbox_root_mupots_output.json'
    with open(output_path, 'w') as f:
        json.dump(pred_save, f)
    print("Test result is saved at " + output_path)


def first_model(streamManagerApi1_fi, intrinsic_fi):
    streamManagerApi_1 = streamManagerApi1_fi
    # 新建一个流管理StreamManager对象并初始化
    ret = streamManagerApi_1.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    with open("pipeline/detection_yolov3.pipeline", "rb") as f:
        pipeline_str = f.read()
    ret = streamManagerApi_1.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # 构建流的输入对象--检测目标
    Data_Input = MxDataInput()
    if os.path.exists(IMG_PATH) != 1:
        print("The test image does not exist.")
    with open(IMG_PATH, 'rb') as f:
        Data_Input.data = f.read()

    streamName = b'detection'
    inPluginId = 0
    # 根据流名将检测目标传入流中
    uniqueId = streamManagerApi_1.SendData(streamName, inPluginId, Data_Input)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()
    keys = [b"mxpi_objectpostprocessor0", b"mxpi_imagecrop0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)
    # 从流中取出对应插件的输出数据
    infer_result = streamManagerApi_1.GetProtobuf(streamName, 0, keyVec)
    if infer_result.size() == 0:
        print("infer_result is null")
        exit()

    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (
            infer_result[0].errorCode))
        exit()

    # mxpi_objectpostprocessor0模型后处理插件输出信息
    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(infer_result[0].messageBuf)
    print(objectList)

    #the second model input value
    k_value_n = []
    ned_bbox = []
    for i in range(0, len(objectList.objectVec)):
        result = objectList.objectVec[i]
        if result.classVec[0].className == "person":
            if result.x0 is not None:
                x0 = result.x0
            else:
                x0 = 0
            if result.y0 is not None:
                y0 = result.y0
            else:
                y0 = 0
            if result.x1 is not None:
                x1 = result.x1
            else:
                x1 = 256
            if result.y1 is not None:
                y1 = result.y1
            else:
                y1 = 256

            original_img = cv2.imread(IMG_PATH)
            original_img_height, original_img_width = original_img.shape[:2]
            bbox_list = []
            bbox_list.append(x0)
            bbox_list.append(y0)
            bbox_list.append(x1-x0)
            bbox_list.append(y1-y0)
            bbox = process_bbox(np.array(bbox_list), original_img_width, original_img_height)
            k_value_n.append(get_k_value(bbox, intrinsic_fi))   
            ned_bbox.append(bbox)
            # mxpi_imagecrop0 图像抠图输出信息
            imagecrop = MxpiDataType.MxpiVisionList()
            imagecrop.ParseFromString(infer_result[1].messageBuf)
            vision_data = imagecrop.visionVec[i].visionData.dataStr
            visionInfo =  imagecrop.visionVec[i].visionInfo
            YUV_BYTES_NU = 3
            YUV_BYTES_DE = 2
            img_yuv = np.frombuffer(vision_data, np.uint8)
            img_bgr = img_yuv.reshape(visionInfo.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo.widthAligned)
            # 用输出原件信息初始化OpenCV图像信息矩阵
            img = cv2.cvtColor(img_bgr, getattr(cv2, "COLOR_YUV2BGR_NV12"))
            cv2.imwrite('output_root_2d_' + str(i) + '.jpg', img)
    return k_value_n, ned_bbox


def second_model(streamManagerApi2, ned_bbox, k_value_n, keypoints_cam_m, keypoints_img_n, intrinsic_fi):
    streamManagerApi_2 = streamManagerApi2
    # 新建一个流管理StreamManager对象并初始化
    ret = streamManagerApi_2.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    # 构建pipeline
    # create streams by pipeline config file
    with open("pipeline/detection_3d.pipeline", "rb") as f:
        pipeline_str = f.read()
    ret = streamManagerApi_2.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    # 构建流的输入对象--检测目标
    d2_img = cv2.imread(IMG_PATH)
    result_root = []
    k_value_n_len = len(k_value_n)
    for ik in range(0, k_value_n_len):
        if os.path.exists('output_root_2d_' + str(ik) + '.jpg') != 1:
            print("The test image does not exist.")
        image1 = cv2.imread('output_root_2d_' + str(ik) + '.jpg')
        image = image1.transpose([2, 0, 1]) # hwc to chw
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)
        image = image.numpy()

        # gen tensor data
        mxpi_tensor_pack_list = MxpiDataType.MxpiTensorPackageList()
        tensor_package_vec = mxpi_tensor_pack_list.tensorPackageVec.add()
        # add object data
        tensorVec_obj = tensor_package_vec.tensorVec.add()
        tensorVec_obj.memType = 1
        tensorVec_obj.deviceId = 0
        tensorVec_obj.tensorDataSize = int(256 * 256 * 3 * 4) # hwc float32
        tensorVec_obj.tensorDataType = 0 # float32
        for i in image.shape:
            tensorVec_obj.tensorShape.append(i)
        tensorVec_obj.dataStr = image.tobytes()
        # add layout data
        list1 = []
        value = k_value_n[ik]
        list1.append(value)
        # 将python的List类型转换为numpy的ndarray
        layout = np.array(list1)
        tensorVec_lay = tensor_package_vec.tensorVec.add()
        tensorVec_lay.memType = 1
        tensorVec_lay.deviceId = 0
        tensorVec_lay.tensorDataSize = int(1) # H*W*C*(float32)
        tensorVec_lay.tensorDataType = 0 # float32
        for i in layout.shape:
            tensorVec_lay.tensorShape.append(i)
        tensorVec_lay.dataStr = layout.tobytes()

        # send data to stream
        protobuf_in = MxProtobufIn()
        protobuf_in.key = b'appsrc0'
        protobuf_in.type = b'MxTools.MxpiTensorPackageList'
        protobuf_in.protobuf = mxpi_tensor_pack_list.SerializeToString()
        protobuf_vec = InProtobufVector()
        protobuf_vec.push_back(protobuf_in)
        stream_name = b'detection3d'
        in_plugin_id = 0
        unique_id = streamManagerApi_2.SendProtobuf(stream_name, in_plugin_id, protobuf_vec)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        # get inference result
        keys = [b"mxpi_tensorinfer1"]
        key_vec = StringVector()
        for key in keys:
            key_vec.push_back(key)

        infer_raw = streamManagerApi_2.GetResult(stream_name, b'appsink0', key_vec)
        print("result.metadata size: ", infer_raw.metadataVec.size())

        infer_result = infer_raw.metadataVec[0]

        if infer_result.errorCode != 0:
            print("GetResult error. errorCode=%d , errMsg=%s" % (
                infer_result.errorCode, infer_result.errMsg))
            exit()

        # convert result
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result.serializedMetadata)
        img1_rgb = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr
                                 , dtype = np.float32)

        input_shape = (256, 256)
        output_shape = (input_shape[0]//4, input_shape[1]//4)
        root_3d = img1_rgb.copy()
        result_root.append(root_3d)
        vis_img = image1.copy()
        # save output in 2D space (x,y: pixel)
        vis_root = np.zeros((2))
        vis_root[0] = root_3d[0] / output_shape[1] * input_shape[1]
        vis_root[1] = root_3d[1] / output_shape[0] * input_shape[0]
        bboxx = ned_bbox[ik]
        viss_root = np.zeros((2))
        viss_root[0] = root_3d[0] / output_shape[1] * bboxx[2] + bboxx[0]
        viss_root[1] = root_3d[1] / output_shape[0] * bboxx[3] + bboxx[1]
        keypoints_img_n[ik][14] = viss_root
        fx = intrinsic_fi[0]
        fy = intrinsic_fi[1]
        cx = intrinsic_fi[2]
        cy = intrinsic_fi[3]
        f = np.array([fx, fy])
        c = np.array([cx, cy])
        visss_root = img1_rgb.copy()
        visss_root[0] = viss_root[0]
        visss_root[1] = viss_root[1]
        pred_root = pixel2cam(visss_root[None, :], f, c)[0]
        keypoints_cam_m[ik][14] = pred_root
        cv2.circle(vis_img, (int(vis_root[0]), int(vis_root[1])), radius=5,
                   color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
        cv2.imwrite('output_root_2d_' + str(ik) + '.jpg', vis_img)
        print('Root joint depth: ' + str(root_3d[2]) + ' mm')
    # visualize 2d poses
    d2_img = vis_keypoints(d2_img, keypoints_img_n, SKELETON)
    cv2.imwrite('output_root_2d_pose.jpg', d2_img)
    # visualize 3d poses
    vis_kps = keypoints_cam_m
    vis_3d_multiple_skeleton(vis_kps, np.ones_like(vis_kps), SKELETON, 'output_pose_3d (x,y,z: camera-centered. mm.)')
    # destroy streams
    streamManagerApi_2.DestroyAllStreams()
    return result_root

if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    Config_File = 'item.ini'
    net_config = read_config(Config_File)
    keypoints_cam, keypoints_img, intrinsic = get_config_data(net_config)
    print(keypoints_cam)
    print(keypoints_img)
    print(intrinsic)
    k_value, nedd_bbox = first_model(streamManagerApi, intrinsic)
    rresult_root = []
    rresult_root = second_model(streamManagerApi, nedd_bbox, k_value, keypoints_cam, keypoints_img, intrinsic)
    evaljosn(nedd_bbox, rresult_root, intrinsic)