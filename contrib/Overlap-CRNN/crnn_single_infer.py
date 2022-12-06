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

import os

import mindx.sdk as sdk
from mindx.sdk.base import Tensor

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = "./models/om_model/crnn.om"
LABEL_DICT_PATH = "./ch_sim_en_digit_symble.txt"
IMAGE_PATH = "./test.jpg"
SAVE_PATH = "./show.jpg"
font = ImageFont.truetype(font='./Ubuntu-Regular.ttf', size=20)
DEVICE_ID = 0
BLANK = 6702


def infer():
    if not os.path.exists(MODEL_PATH):
        print("The input model path is empty!!!")
        print("plz place the model in ./Overlap-CRNN/models/om_model/")
        exit()
    crnn_model = sdk.model(MODEL_PATH, DEVICE_ID)
    if not os.path.exists(IMAGE_PATH):
        print("The input image path is empty!!!")
        print("plz place the image in ./Overlap-CRNN/")
        exit()
    img = load_img_data(IMAGE_PATH)
    output = crnn_model.infer(img)
    output[0].to_host()
    output[0] = np.array(output[0])
    result = Ctc_Post_Process(y_pred=output[0], blank=BLANK)
    result = result[0]
    img_show(IMAGE_PATH, result)
    print("predict text: ", result)


def load_img_data(image_name):
    if cv2.imread(image_name) is None:
        print("=============!Error!================")
        print("The input image is empty, plz check out!")
        print("====================================")
        exit()
    else:
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


def resize(img, height, width):
    img = sdk.dvpp.resize(img, height, width)
    img = img.get_tensor()
    return img


def arr2char(inputs):
    string = ""
    for num in inputs:
        if num < BLANK:
            string += g_labelDict[num]
    return string


def Ctc_Post_Process(y_pred, blank):
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


def img_show(img, pred):
    img = Image.open(img)
    canvas = Image.new('RGB', (img.size[0], int(img.size[1] * 1.5)),
                       (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    label_size = draw.textsize(pred, font)
    text_origin = np.array(
        [int((img.size[0] - label_size[0]) / 2), img.size[1] + 1])
    draw.text(text_origin, pred, fill='red', font=font)
    canvas.paste(img, (0, 0))
    canvas.save(SAVE_PATH)
    canvas.show()


try:
    g_labelDict = ""
    if not os.path.exists(LABEL_DICT_PATH):
        print("The input dictionary path is empty!!!")
        print("plz place the model in ./Overlap-CRNN/")
        exit()
    r_dict = open(LABEL_DICT_PATH, 'r')
    g_labelDict = r_dict.read().splitlines()  # 将字典读入一个str中
    infer()

except Exception as e:
    print(e)
