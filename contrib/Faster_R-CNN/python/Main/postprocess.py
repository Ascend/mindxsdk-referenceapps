# Copyright 2022 Huawei Technologies Co., Ltd
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
# ============================================================================

import os
import stat
import json
import shutil
import numpy as np
import cv2 as cv
import tqdm
import matplotlib.pyplot as plt


def json_to_txt(infer_result_path, savetxt_path):
    if os.path.exists(savetxt_path):
        shutil.rmtree(savetxt_path)
    os.mkdir(savetxt_path)
    files = os.listdir(infer_result_path)
    for file in files:
        if file.endswith(".json"):
            json_path = os.path.join(infer_result_path, file)
            with open(json_path, 'r') as fp:
                result = json.loads(fp.read())
            if result:
                data = result.get("MxpiObject")
                txt_file = file.split(".")[0] + ".txt"
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                modes = stat.S_IWUSR | stat.S_IRUSR
                with os.fdopen(os.open(os.path.join(savetxt_path, txt_file), flags, modes), 'w') as f:
                    if file.split('_')[0] == "W0003":
                        temp = int(file.split("_")[2]) - 600
                    else:
                        temp = int(file.split("_")[1]) - 600
                    for bbox in data:
                        class_vec = bbox.get("classVec")[0]
                        class_id = int(class_vec["classId"])
                        confidence = class_vec.get("confidence")
                        xmin = bbox["x0"]
                        ymin = bbox["y0"]
                        xmax = bbox["x1"]
                        ymax = bbox["y1"]
                        if xmax - xmin >= 5 and ymax - ymin >= 5:
                            f.write(
                                str(xmin + temp) + ',' + str(ymin) + ',' + str(xmax + temp) + ',' + str(
                                    ymax) + ',' + str(
                                    round(confidence, 2)) + ',' + str(class_id) + '\n')


def hebing_txt(txt_path, save_txt_path, remove_txt_path, cut_path):
    if not os.path.exists(save_txt_path):
        os.makedirs(save_txt_path)
    if not os.path.exists(remove_txt_path):
        os.makedirs(remove_txt_path)
    fileroot = os.listdir(save_txt_path)
    remove_list = os.listdir(remove_txt_path)
    for filename in remove_list:
        os.remove(os.path.join(remove_txt_path, filename))
    for filename in fileroot:
        os.remove(os.path.join(save_txt_path, filename))
    data = []
    for file in os.listdir(cut_path):
        data.append(file.split(".")[0])
    txt_list = os.listdir(txt_path)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    for image in data:
        fw = os.fdopen(os.open(os.path.join(save_txt_path, image + '.txt'), flags, modes), 'w')
        for txtfile in txt_list:
            if image.split('_')[0] == "W0003":
                if image.split('_')[1] == txtfile.split('_')[1]:
                    for line in open(os.path.join(txt_path, txtfile), "r"):
                        fw.write(line)
            else:
                if image.split('_')[0] == txtfile.split('_')[0]:
                    for line in open(os.path.join(txt_path, txtfile), "r"):
                        fw.write(line)
        fw.close()

    fileroot = os.listdir(save_txt_path)
    for file in fileroot:
        oldname = os.path.join(save_txt_path, file)
        newname = os.path.join(remove_txt_path, file)
        shutil.copyfile(oldname, newname)


def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return keep


def plot_bbox(dets, c='k'):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    plt.plot([x1, x2], [y1, y1], c)
    plt.plot([x1, x1], [y1, y2], c)
    plt.plot([x1, x2], [y2, y2], c)
    plt.plot([x2, x2], [y1, y2], c)
    plt.title(" nms")


def nms_box(image_path, image_save_path, txt_path, thresh, obj_list):
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    remove_list = os.listdir(image_save_path)
    for filename in remove_list:
        os.remove(os.path.join(image_save_path, filename))
    txt_list = os.listdir(txt_path)
    for txtfile in tqdm.tqdm(txt_list):
        boxes = np.loadtxt(os.path.join(txt_path, txtfile), dtype=np.float32,
                           delimiter=',')
        if boxes.size > 5:
            if os.path.exists(os.path.join(txt_path, txtfile)):
                os.remove(os.path.join(txt_path, txtfile))
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            modes = stat.S_IWUSR | stat.S_IRUSR
            fw = os.fdopen(os.open(os.path.join(txt_path, txtfile), flags, modes), 'w')

            keep = py_cpu_nms(boxes, thresh=thresh)

            img = cv.imread(os.path.join(image_path, txtfile[:-3] + 'jpg'), 0)
            for label in boxes[keep]:
                fw.write(str(int(label[0])) + ',' + str(int(label[1])) + ',' + str(int(label[2])) + ',' + str(
                    int(label[3])) + ',' + str(round((label[4]), 2)) + ',' + str(int(label[5])) + '\n')
                x_min = int(label[0])
                y_min = int(label[1])
                x_max = int(label[2])
                y_max = int(label[3])

                color = (0, 0, 255)
                if x_max - x_min >= 5 and y_max - y_min >= 5:
                    cv.rectangle(img, (x_min, y_min), (x_max, y_max), color, 1)
                    font = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(img, (obj_list[int(label[5])] + str(round((label[4]), 2))),
                               (x_min, y_min - 7), font, 0.4, (6, 230, 230), 1)
            cv.imwrite(os.path.join(image_save_path, txtfile[:-3] + 'jpg'), img)
            fw.close()


def post_process():
    infer_result_path = "../data/test/infer_result"
    txt_save_path = "../data/test/img_txt"
    json_to_txt(infer_result_path, txt_save_path)

    txt_path = "../data/test/img_txt"
    all_txt_path = "../data/test/img_huizong_txt"
    nms_txt_path = "../data/test/img_huizong_txt_nms"
    cut_path = "../data/test/cut"
    hebing_txt(txt_path, all_txt_path, nms_txt_path, cut_path)

    cut_path = "../data/test/cut"
    image_save_path = "../data/test/draw_result"
    nms_txt_path = "../data/test/img_huizong_txt_nms"
    obj_lists = ['qikong', 'liewen']
    nms_box(cut_path, image_save_path, nms_txt_path, thresh=0.1, obj_list=obj_lists)


if __name__ == '__main__':
    post_process()