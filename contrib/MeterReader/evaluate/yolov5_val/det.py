# !/usr/bin/env python
# coding=utf-8

# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

'''
1.mapDet.py:得到验证集中数据的预测结果，并且将结果保存成txt，以”图片名.txt“命名，txt中数据格式为：
    <class_id> <x_center> <y_center> <width> <height>

'''

import json
import os
import stat
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


PIPELINE = '../pipeline/yolov5/det.pipeline'
FILEPATH = '../evaluate/yolov5_val/det_val_data/det_val_img'
SAVE_PATH = '../evaluate/yolov5_val/det_val_data/det_sdk_img/'
SAVE_TXT = '../evaluate/yolov5_val/det_val_data/det_sdk_txt/'
MODES = stat.S_IWUSR | stat.S_IRUSR
FLAGS = os.O_WRONLY | os.O_CREAT


class DetPostProcessors:
    def __init__(self):
        self.pred = None
        self.source = None
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False
        self.max_det = 1000
        self.multi_label = False
        self.save_path = 'det_res.jpg'
        self.save_txt = ''

    @staticmethod
    def xyxy2xywh(x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right

        y = np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height

        return y

    @staticmethod
    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

        return y

    @staticmethod
    def box_iou(box1, box2):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (np.min(box1[:, None, 2:], box2[:, 2:]) - np.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    @staticmethod
    def new_nms(bboxes, scores, threshold=0.5):
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)

        # 从大到小对应的的索引
        order = scores.argsort()[::-1]

        # 记录输出的bbox
        keep = []
        while order.size > 0:
            i = order[0]
            # 记录本轮最大的score对应的index
            keep.append(i)

            if order.size == 1:
                break

            # 计算当前bbox与剩余的bbox之间的IoU
            # 计算IoU需要两个bbox中最大左上角的坐标点和最小右下角的坐标点
            # 即重合区域的左上角坐标点和右下角坐标点
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # 如果两个bbox之间没有重合, 那么有可能出现负值
            w = np.maximum(0.0, (xx2 - xx1))
            h = np.maximum(0.0, (yy2 - yy1))
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 删除IoU大于指定阈值的bbox(重合度高), 保留小于指定阈值的bbox
            ids = np.where(iou <= threshold)[0]
            # 因为ids表示剩余的bbox的索引长度
            # +1恢复到order的长度
            order = order[ids + 1]
        keep = np.array(keep)

        return keep

    @staticmethod
    def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        DetPostProcessors.clip_coords(coords, img0_shape)
        return coords

    @staticmethod
    def clip_coords(boxes, shape):
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    @staticmethod
    def non_max_suppression(prediction, conf_thres, iou_thres, classes, agnostic, multi_label,
                            labels=(), max_det=300):

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        output = [np.zeros((0, 6))] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                # torch.zeros返回一个由标量0填充的张量，它的形状由size决定
                v = np.zeros((len(lb), nc + 5))
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = np.concatenate((x, v), axis=0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            box = DetPostProcessors.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].float()), axis=1)
            else:  # best class only

                conf = np.max(x[:, 5:], axis=1)
                j = np.argmax(x[:, 5:], axis=1)
                conf = np.array(conf)[None].T
                j = np.array(j)[None].T

                temp_x = np.concatenate((box, conf, j), axis=1)
                x = np.squeeze(temp_x)

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == np(classes)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

            i = DetPostProcessors.new_nms(boxes, scores)  # NMS

            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = DetPostProcessors.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights # merged boxes
                x[i, :4] = np.matmul(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]

        return output

    @staticmethod
    def xyxy2xywh(x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        z = [0, 0, 0, 0]
        gn = [1920, 1080, 1920, 1080]
        z[0] = (x[0] + x[2]) / 2  # x center
        z[1] = (x[1] + x[3]) / 2  # y center
        z[2] = x[2] - x[0]  # width
        z[3] = x[3] - x[1]  # height

        temp_xyxy = (np.array(z)[None]) / gn
        xywh = np.squeeze(temp_xyxy, axis=0)
        x1 = str(xywh[0])[:8]
        x2 = str(xywh[1])[:8]
        x3 = str(xywh[2])[:8]
        x4 = str(xywh[3])[:8]
        temp_line = "0" + " " + x1 + " " + x2 + " " + x3 + " " + x4 + " "
        return temp_line

    def run(self):
        # NMS 非极大值抑制
        pred_result_in = DetPostProcessors.non_max_suppression(self.pred, self.conf_thres, self.iou_thres, self.classes,
                                                            self.agnostic_nms, self.multi_label)

        line = ''
        file_name = self.source.split("/")[-1][:-4]
        # Process predictions
        for i, det in enumerate(pred_result_in):  # per image
            im0 = cv2.imread(self.source)
            if len(det):
                # [1024, 576]是图片根据模型输入resize后的尺寸
                res = DetPostProcessors.scale_coords([1024, 576], det[:, :4], im0.shape).round()

                for meter in res:
                    x = [int(meter[0]), int(meter[1]), int(meter[2]), int(meter[3])]
                    temp_line = DetPostProcessors.xyxy2xywh(x)

                    cv2.rectangle(im0, (int(meter[0]), int(meter[1])), (int(meter[2]), int(meter[3])), (0, 255, 0), 2)
                    res_img_path = (self.save_path + file_name + '.jpg').replace("\\", "/")
                    cv2.imwrite(res_img_path, im0)
                    line += temp_line + str(round(det[:, :5][-1][-1], 6)) + "\n"

        lable_path = (self.save_txt + "/" + file_name + '.txt').replace("\\", "/")

        with os.fdopen(os.open(lable_path, os.O_WRONLY | os.O_CREAT, MODES), 'w') as lable:
            lable.write(line)


if __name__ == '__main__':
    # 改写pipeline里面的model路径

    # file_object = open(pipeline_path, 'r')

    # content = json.load(file_object)
    # modelPath = model_path
    # content['detection']['mxpi_tensorinfer0']['props']['modelPath'] = modelPath

    steammanager_api = StreamManagerApi()
    # init stream manager
    ret = steammanager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    MODES = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(PIPELINE, os.O_RDONLY, MODES), 'rb') as f:
        pipeline_str = f.read()
    ret = steammanager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    dataInput = MxDataInput()
    # It is best to use absolute path

    if os.path.exists(FILEPATH) != 1:
        print("The test image does not exist. Exit.")
        exit()
    imgs = os.listdir(FILEPATH)
    DetPostProcessors = DetPostProcessors()
    for img in imgs:
        img_path = os.path.join(FILEPATH, img)
        with os.fdopen(os.open(img_path, os.O_RDONLY, MODES), 'rb') as f:
            dataInput.data = f.read()
        STEAMNAME = b'detection'
        INPLUGINID = 0
        uniqueId = steammanager_api.SendData(STEAMNAME, INPLUGINID, dataInput)
        if uniqueId < 0:
            print("Failed to send data to stream.")
            exit()

        keys = [b"mxpi_tensorinfer0"]
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)

        # 从流中取出对应插件的输出数据
        infer = steammanager_api.GetResult(STEAMNAME, b'appsink0', keyVec)
        if (infer.metadataVec.size() == 0):
            print("Get no data from stream !")
            exit()
        print("result.metadata size: ", infer.metadataVec.size())
        infer_result = infer.metadataVec[0]
        if infer_result.errorCode != 0:
            print("GetResult error. errorCode=%d , errMsg=%s" % (infer_result.errorCode, infer_result.errMsg))
            exit()

        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result.serializedMetadata)

        pred_result = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        pred_result.resize(1, 36288, 6)
        np.save(r"img_pred.npy", pred_result)

        pred_result = np.load(r"img_pred.npy")

        DetPostProcessors.source = img_path
        DetPostProcessors.pred = pred_result
        DetPostProcessors.save_path = SAVE_PATH
        DetPostProcessors.save_txt = SAVE_TXT
        DetPostProcessors.run()

        print("It is finish!")

    # destroy streams
    steammanager_api.DestroyAllStreams()
