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
    imgs, labels = data_loader()
    results = []
    for num in range(len(imgs)):
        img_path = imgs[num]
        label = labels[num]
        image_tensor_list = load_img_data(img_path)
        output = crnn_model.infer(image_tensor_list)
        output[0].to_host()
        output[0] = np.array(output[0])
        result = CTCPostProcess(y_pred=output[0], blank=BLANK)
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
    """
    使用外部数据作为tensor时务必使用to_device进行转移，缺失该步骤会导致输出结果异常，RC3以上版本已修复
    ！！！如使用了numpy.transpose等改变数据内存形状的操作后，需要使用numpy.ascontiguousarray对内存进行重新排序成连续的
    如使用非图像数据，也是转为numpy.ndarray数据类型再进行Tensor转换，使用{tensor_data} = Tensor({numpy_data})方式
    """
    return image_tensor_list


def arr2char(inputs):
    string = ""
    for input in inputs:
        if input < BLANK:
            string += label_dict[input]
    return string


def CTCPostProcess(y_pred, blank):
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
    annotation_path = os.path.join(IMAGE_PATH, 'mask_annotation.txt')
    imgs = []
    texts = []
    with open(annotation_path, 'r') as f:
        datas = json.loads(f.read())
    print(len(datas))

    for _, data in enumerate(datas):
        for _, text_data in enumerate(data['texts']):
            imgs.append(os.path.join(IMAGE_PATH, text_data['mask']))
            texts.append(text_data['label'])
    return imgs, texts


def data_loader():
    annotation_path = os.path.join(IMAGE_PATH, 'mask_annotation.txt')
    imgs = []
    texts = []
    f = open(annotation_path, 'r')
    datas = f.read().splitlines()

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
    label_dict = ""
    f = open(LABEL_DICT_PATH, 'r')
    label_dict = f.read().splitlines()
    print('label len:', len(label_dict))
    infer()

except Exception as e:
    print(e)
