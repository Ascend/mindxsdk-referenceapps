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
import stat
import math
import configparser
import natsort
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxProtobufIn, MxDataInput, StringVector, InProtobufVector

NAME_MAX_LEN = 100;
BBOX_REAL = [2000, 2000]
SKELETON = ((0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11),
            (8, 9), (9, 10), (11, 12), (12, 13), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7))


def read_config(config_fname):
    curpath = os.path.dirname(os.path.realpath(__file__))
    cfgpath = os.path.join(curpath, config_fname)

    conf = configparser.ConfigParser()
    conf.read(cfgpath, encoding="utf-8")

    return conf


def get_config_data(net_param):
    items_intrinsic = net_param.items("intrinsic")
    lem_items_intrinsic = len(items_intrinsic)
    for i in range(0, lem_items_intrinsic):
        boxes_str = net_param.get("intrinsic", items_intrinsic[i][0])
        intrinsic_m = json.loads(boxes_str)

    keypoints_cam_n = []
    items_keypoints_cam = net_param.items("keypoints_cam")
    lem_items_keypoints_cam = len(items_keypoints_cam)
    for i in range(0, lem_items_keypoints_cam):
        boxes_str = net_param.get("keypoints_cam", items_keypoints_cam[i][0])
        keypoints_cam_n.append(json.loads(boxes_str))

    keypoints_img_n = []
    items_keypoints_img = net_param.items("keypoints_img")
    lem_items_keypoints_img = len(items_keypoints_img)
    for i in range(0, lem_items_keypoints_img):
        boxes_str = net_param.get("keypoints_img", items_keypoints_img[i][0])
        keypoints_img_n.append(json.loads(boxes_str))

    if len(keypoints_cam_n) != len(keypoints_img_n):
        print("keypoints_cam_n and keypoints_img_n is error")
        exit()

    image_iid = net_param.items("image_id")
    len_image_id = len(image_iid)
    for i in range(0, len_image_id):
        boxes_str = net_param.get("image_id", image_iid[i][0])
        image_id_m = json.loads(boxes_str)
    res_list = []
    res_list.append(keypoints_cam_n)
    res_list.append(keypoints_img_n)
    return res_list, intrinsic_m, image_id_m


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


def vis_3d_multiple_skeleton(kpt_3d, kpt_3d_vis, kps_lines, img_3d_path, filename=None):
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
    img_3d_path = img_3d_path[:-8] + '3d_pose.png'
    plt.savefig(img_3d_path)


def get_k_value(bbox_n, intrinsic_nn):
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
    if w * h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    input_shape = [256, 256]
    aspect_ratio = input_shape[1] / input_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * 1.25
    bbox[3] = h * 1.25
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.
    return bbox


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return cam_coord


def evaljosn1(rebboxd, root3dd, intrinsic_nnd, image_iddd, pred_save, countresultev):
    len_root3dd = len(root3dd)
    for n in range(0, len_root3dd):
        root3d = root3dd[n]
        rebbox = rebboxd[n]
        intrinsic_nn = intrinsic_nnd[n]
        image_idd = image_iddd[n]
        print('make Evaluation json ' + str(n) + ' start...')
        sample_num = countresultev[n]
        fx = intrinsic_nn[0]
        fy = intrinsic_nn[1]
        cx = intrinsic_nn[2]
        cy = intrinsic_nn[3]
        f = np.array([fx, fy])
        c = np.array([cx, cy])
        output_shape = [256 // 4, 256 // 4]
        for nn in range(sample_num):
            image_id = image_idd
            bbox = rebbox[nn]
            score = 1
            # restore coordinates to original space
            pred_root = root3d[nn].copy()
            pred_root[0] = pred_root[0] / output_shape[1] * bbox[2] + bbox[0]
            pred_root[1] = pred_root[1] / output_shape[0] * bbox[3] + bbox[1]

            # back project to camera coordinate system
            pred_root = pixel2cam(pred_root[None, :], f, c)[0]

            pred_save.append(
                {'image_id': image_id, 'root_cam': pred_root.tolist(), 'bbox': bbox.tolist(), 'score': score})


def first_model(img_pathf, intrinsic_fif):
    stream_one = StreamManagerApi()
    # 新建一个流管理StreamManager对象并初始化
    ret = stream_one.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    with open("pipeline/detection_yolov3_crop.pipeline", "rb") as f:
        pipeline_str = f.read()
    ret = stream_one.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    k_value_n_list = []
    ned_bbox_list = []
    image_save_list = []
    # 构建流的输入对象--检测目标
    data_input = MxDataInput()
    len_img_pathf = len(img_pathf)
    for n in range(0, len_img_pathf):
        img_pathff = img_pathf[n]
        intrinsic_fi = intrinsic_fif[n]
        if os.path.exists(img_pathff) != 1:
            print("The test image does not exist.")
        with open(img_pathff, 'rb') as f:
            data_input.data = f.read()

        stream_name = b'detection'
        plugin_id = 0
        # 根据流名将检测目标传入流中
        unique_id = stream_one.SendData(stream_name, plugin_id, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()

        keys = [b"mxpi_objectpostprocessor0", b"mxpi_imagecrop0"]
        key_vec = StringVector()
        for key in keys:
            key_vec.push_back(key)
        # 从流中取出对应插件的输出数据
        infer_result = stream_one.GetProtobuf(stream_name, 0, key_vec)
        if infer_result.size() == 0:
            print("infer_result is null")
            exit()

        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (
                infer_result[0].errorCode))
            exit()


        # mxpi_objectpostprocessor0模型后处理插件输出信息
        object_list = MxpiDataType.MxpiObjectList()
        object_list.ParseFromString(infer_result[0].messageBuf)
        print(object_list)

        if infer_result.size() != 2:
            print("crop is error, maybe size is smaller than 20*20")
            continue

        # the second model input value
        k_value_n = []
        ned_bbox = []
        image_save_n = []
        for i in range(0, len(object_list.objectVec)):
            result = object_list.objectVec[i]
            original_img = cv2.imread(img_pathff)
            original_img_height, original_img_width = original_img.shape[:2]
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
                    x1 = original_img_width
                if result.y1 is not None:
                    y1 = result.y1
                else:
                    y1 = original_img_height

                bbox_list = []
                bbox_list.append(x0)
                bbox_list.append(y0)
                bbox_list.append(x1 - x0)
                bbox_list.append(y1 - y0)
                bbox = process_bbox(np.array(bbox_list), original_img_width, original_img_height)
                k_value_n.append(get_k_value(bbox, intrinsic_fi))
                ned_bbox.append(bbox)
                # mxpi_imagecrop0 图像抠图输出信息
                imagecrop = MxpiDataType.MxpiVisionList()
                imagecrop.ParseFromString(infer_result[1].messageBuf)
                vision_data = imagecrop.visionVec[i].visionData.dataStr
                vision_info = imagecrop.visionVec[i].visionInfo
                bytes_nu = 3
                bytes_de= 2
                img_yuv = np.frombuffer(vision_data, np.uint8)
                img_bgr = img_yuv.reshape(vision_info.heightAligned * bytes_nu // bytes_de,
                                          vision_info.widthAligned)
                # 用输出原件信息初始化OpenCV图像信息矩阵
                img = cv2.cvtColor(img_bgr, getattr(cv2, "COLOR_YUV2BGR_NV12"))
                image_save = img_pathff[:-4] + 'output_root_2d_' + str(i) + '.jpg'
                image_save_n.append(image_save)
                cv2.imwrite(image_save, img)
        k_value_n_list.append(k_value_n)
        ned_bbox_list.append(ned_bbox)
        image_save_list.append(image_save_n)
    return k_value_n_list, ned_bbox_list, image_save_list


def second_model(ned_bboxs, k_value_ns, keypoints_cam_ms, keypoints_img_ns, intrinsic_fis, image_saves, img_pathcs):
    stream_two = StreamManagerApi()
    # 新建一个流管理StreamManager对象并初始化
    ret = stream_two.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    # 构建pipeline
    # create streams by pipeline config file
    with open("pipeline/detection_3d.pipeline", "rb") as f:
        pipeline_str = f.read()
    ret = stream_two.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    result_root_list = []
    countresult = []
    len_image_saves = len(image_saves)
    for n in range(0, len_image_saves):
        d2_img = cv2.imread(img_pathcs[n])
        ned_bbox = ned_bboxs[n]
        keypoints_cam_m = keypoints_cam_ms[n]
        keypoints_img_n = keypoints_img_ns[n]
        intrinsic_fi = intrinsic_fis[n]
        result_root = []
        k_value_n = k_value_ns[n]
        piclen = len(keypoints_cam_m)
        if len(keypoints_cam_m) > len(k_value_n):
            piclen = len(k_value_n)
        # 构建流的输入对象--检测目标
        countresult.append(piclen)
        for ik in range(0, piclen):
            img_paths = image_saves[n][ik]
            if os.path.exists(img_paths) != 1:
                print("The test image does not exist.")
            image1 = cv2.imread(img_paths)
            image = image1.transpose([2, 0, 1])  # hwc to chw
            image = torch.from_numpy(image).float()
            image = image.unsqueeze(0)
            image = image.numpy()

            # gen tensor data
            mxpi_tensor_pack_list = MxpiDataType.MxpiTensorPackageList()
            tensor_package_vec = mxpi_tensor_pack_list.tensorPackageVec.add()
            # add object data
            tensor_obj = tensor_package_vec.tensorVec.add()
            tensor_obj.memType = 1
            tensor_obj.deviceId = 0
            tensor_obj.tensorDataSize = int(256 * 256 * 3 * 4)  # hwc float32
            tensor_obj.tensorDataType = 0  # float32
            for i in image.shape:
                tensor_obj.tensorShape.append(i)
            tensor_obj.dataStr = image.tobytes()
            # add layout data
            list1 = []
            value = k_value_n[ik]
            list1.append(value)
            # 将python的List类型转换为numpy的ndarray
            layout = np.array(list1)
            tensor_lay = tensor_package_vec.tensorVec.add()
            tensor_lay.memType = 1
            tensor_lay.deviceId = 0
            tensor_lay.tensorDataSize = int(1)  # H*W*C*(float32)
            tensor_lay.tensorDataType = 0  # float32
            for i in layout.shape:
                tensor_lay.tensorShape.append(i)
            tensor_lay.dataStr = layout.tobytes()

            # send data to stream
            protobuf_in = MxProtobufIn()
            protobuf_in.key = b'appsrc0'
            protobuf_in.type = b'MxTools.MxpiTensorPackageList'
            protobuf_in.protobuf = mxpi_tensor_pack_list.SerializeToString()
            protobuf_vec = InProtobufVector()
            protobuf_vec.push_back(protobuf_in)
            stream_name = b'detection3d'
            in_plugin_id = 0
            unique_id = stream_two.SendProtobuf(stream_name, in_plugin_id, protobuf_vec)
            if unique_id < 0:
                print("Failed to send data to stream.")
                exit()
            # get inference result
            keys = [b"mxpi_tensorinfer1"]
            key_vec = StringVector()
            for key in keys:
                key_vec.push_back(key)

            infer_raw = stream_two.GetResult(stream_name, b'appsink0', key_vec)
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
                                     , dtype=np.float32)

            input_shape = (256, 256)
            output_shape = (input_shape[0] // 4, input_shape[1] // 4)
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
            min_dist_re = 999999
            ik_item = 0
            for ikm in range(0, piclen):
                dist_re = math.sqrt((keypoints_img_n[ikm][14][0] - viss_root[0]) ** 2)
                if dist_re < min_dist_re:
                    min_dist_re = dist_re
                    ik_item = ikm
                else:
                    ik_item = ik_item

            keypoints_img_n[ik_item][14] = viss_root
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
            keypoints_cam_m[ik_item][14] = pred_root
            cv2.circle(vis_img, (int(vis_root[0]), int(vis_root[1])), radius=5,
                       color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.imwrite(img_paths, vis_img)
            print('Root joint depth: ' + str(root_3d[2]) + ' mm')
        # visualize 2d poses
        d2_img = vis_keypoints(d2_img, keypoints_img_n, SKELETON)
        img_2d_paths = img_paths[:-8] + '2d_pose.jpg'
        cv2.imwrite(img_2d_paths, d2_img)
        # visualize 3d poses
        vis_kps = keypoints_cam_m
        vis_3d_multiple_skeleton(vis_kps, np.ones_like(vis_kps), SKELETON, img_paths,
                                 'output_pose_3d (x,y,z: camera-centered. mm.)')
        result_root_list.append(result_root)
    # destroy streams
    stream_two.DestroyAllStreams()
    return result_root_list, countresult


def get_data_path(data_path):
    t = []
    for path, dir1, filelist in os.walk(data_path):
        for filename in filelist:
            if filename.endswith(str('.jpg')):
                temp = list(map(str, path))
                if temp[-1] == '/':
                    temp = temp[:-1]
                temp = ''.join(temp)
                t.append(temp + '/' + filename)

    t1 = []
    for path, dir2, filelist in os.walk(data_path):
        for filename in filelist:
            if filename.endswith(str('.ini')):
                temp = list(map(str, path))
                if temp[-1] == '/':
                    temp = temp[:-1]
                temp = ''.join(temp)
                t1.append(temp + '/' + filename)  # t1空的
    im = natsort.natsorted(list(set(t).difference(set(t1))))
    gt = natsort.natsorted(t1)
    return im, gt


if __name__ == '__main__':
    PIC_PATH = './pic'
    IMG_PATHC, Config_FileC = get_data_path(PIC_PATH)
    if len(IMG_PATHC) != len(Config_FileC):
        print("IMG_PATHC and Config_FileC count is error")
        exit()
    keypoints_cam_list = []
    keypoints_img_list = []
    intrinsic_list = []
    image_id_list = []
    len_config_filec = len(Config_FileC)
    for nu in range(0, len_config_filec):
        Config_File = Config_FileC[nu]
        net_config = read_config(Config_File)
        resnei_list, intrinsic, iimage_id = get_config_data(net_config)
        keypoints_cam = resnei_list[0]
        keypoints_img = resnei_list[1]
        keypoints_cam_list.append(keypoints_cam)
        keypoints_img_list.append(keypoints_img)
        intrinsic_list.append(intrinsic)
        image_id_list.append(iimage_id)

    k_value, nedd_bbox, image_savef = first_model(IMG_PATHC, intrinsic_list)

    rresult_root, countresultw = second_model(nedd_bbox, k_value, keypoints_cam_list, keypoints_img_list,
                                              intrinsic_list, image_savef, IMG_PATHC)

    pred_savee = []
    evaljosn1(nedd_bbox, rresult_root, intrinsic_list, image_id_list, pred_savee, countresultw)
    OUTPUT_PATHH = 'bbox_root_mupots_output.json'
    FLAG_NUM = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    MODE_NUM = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(OUTPUT_PATHH, FLAG_NUM, MODE_NUM), 'w') as fw:
        json.dump(pred_savee, fw)
    print("Test result is saved at " + OUTPUT_PATHH)