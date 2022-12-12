# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import mindx.sdk as sdk
from mindx.sdk.base import Tensor

import numpy as np
import cv2
from PIL import Image

MODEL_PATH = "./models/om_model/crnn.om"
LABEL_DICT_PATH = "./ch_sim_en_digit_symble.txt"
IMAGE_PATH = "./dataset"
DEVICE_ID = 0

BLANK = 6702


def infer():
    crnn_model = sdk.model(MODEL_PATH, DEVICE_ID)
    imgs, labels = json_data_loader()
    results = []
    for i, img in enumerate(imgs):
        img_path = img
        label = labels[i]
        image_tensor_list = load_img_data(img_path)
        output = crnn_model.infer(image_tensor_list)
        output[0].to_host()
        output[0] = np.array(output[0])
        result = ctc_post_process(y_pred=output[0], blank=BLANK)
        result = result[0]
        results.append(result)
    get_acc(results, labels)


def load_img_data(image_name):
    image_name = os.path.join(IMAGE_PATH, image_name)
    im = Image.open(image_name)

    # rgb->bgr
    im = im.convert("RGB")
    r, g, b = im.split()
    im = Image.merge("RGB", (b, g, r))

    # resize
    im = im.resize((112, 32))  # (w,h)
    im = np.array(im)

    # normalize
    mean = np.array([127.5, 127.5, 127.5], dtype=np.float32)
    std = np.array([127.5, 127.5, 127.5], dtype=np.float32)
    img = im.copy().astype(np.float32)
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace

    # HWC-> CHW
    img = img.transpose(2, 0, 1)
    resize_img = img

    # add batch dim
    resize_img = np.expand_dims(resize_img, 0)

    resize_img = np.ascontiguousarray(resize_img)
    image_tensor = Tensor(resize_img)  # 推理前需要转换为tensor的List，使用Tensor类来构建。
    image_tensor.to_device(DEVICE_ID)  # !!!!!重要，需要转移至device侧，该函数单独执行
    image_tensor_list = [image_tensor]  # 推理前需要转换为tensor的List
    return image_tensor_list


def arr2char(inputs):
    string = ""
    for num in inputs:
        if num < BLANK:
            string += LABEL_DICT[num]
    return string


def ctc_post_process(y_pred, blank):
    indices = []
    seq_len, batch_size, _ = y_pred.shape
    indices = y_pred.argmax(axis=2)
    lens = [seq_len] * batch_size
    pred_labels = []
    for i in range(batch_size):
        idx = indices[:, i]
        last_idx = blank
        pred_label = []
        for j in range(lens[i]):
            cur_idx = idx[j]
            if cur_idx not in [last_idx, blank]:
                pred_label.append(cur_idx)
            last_idx = cur_idx
        pred_labels.append(pred_label)
    str_results = []
    for pred in pred_labels:
        pred = arr2char(pred)
        str_results.append(pred)
    return str_results


def json_data_loader():
    annotation_path = os.path.join(IMAGE_PATH, 'annotation.json')
    imgs = []
    texts = []
    with open(annotation_path, 'r') as r_annotation:
        datas = json.loads(r_annotation.read())
    print(len(datas))

    for _, data in enumerate(datas):
        for _, text_data in enumerate(data['texts']):
            imgs.append(text_data['mask'])
            texts.append(text_data['label'])
    return imgs, texts


def data_loader():
    annotation_path = os.path.join(IMAGE_PATH, 'mask_annotation.txt')
    imgs = []
    texts = []
    r_annotation = open(annotation_path, 'r')
    datas = r_annotation.read().splitlines()

    for data in datas:
        img, text = data.split('\t')
        imgs.append(img)
        texts.append(text)
    return imgs, texts


def get_acc(pred_labels, gt_labels):
    true_num = 0
    total_num = len(pred_labels)
    for num in range(total_num):
        if (pred_labels[num].lower() == gt_labels[num].lower()):
            true_num += 1
        else:
            print("pred_label:{}, gt_label:{}".format(pred_labels[num].lower(),
                                                      gt_labels[num].lower()))
    print("==============================")
    print("精度测试结果如下：")
    print("total number:", total_num)
    print("true number:", true_num)
    print("accuracy_rate %.2f" % (true_num / total_num * 100) + '%')
    print("==============================")


try:
    LABEL_DICT = ""
    f = open(LABEL_DICT_PATH, 'r')
    LABEL_DICT = f.read().splitlines()
    print('label len:', len(LABEL_DICT))
    infer()

except Exception as e:
    print(e)
