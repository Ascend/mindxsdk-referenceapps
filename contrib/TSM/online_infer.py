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

import os
import time
import subprocess
import threading
import shutil
from multiprocessing import Manager
from multiprocessing import Process
import numpy as np
from PIL import Image, ImageOps
import mindx.sdk as sdk

REFINE_OUTPUT = True
FILE_PATH = "./model/jester.om"
DEVICE_ID = 0

if not os.path.exists('./image'):
    os.makedirs('./image')
else:
    shutil.rmtree('./image')
    os.makedirs('./image')   
    
catigories = [
    "Doing other things",  # 0
    "Drumming Fingers",  # 1
    "No gesture",  # 2
    "Pulling Hand In",  # 3
    "Pulling Two Fingers In",  # 4
    "Pushing Hand Away",  # 5
    "Pushing Two Fingers Away",  # 6
    "Rolling Hand Backward",  # 7
    "Rolling Hand Forward",  # 8
    "Shaking Hand",  # 9
    "Sliding Two Fingers Down",  # 10
    "Sliding Two Fingers Left",  # 11
    "Sliding Two Fingers Right",  # 12
    "Sliding Two Fingers Up",  # 13
    "Stop Sign",  # 14
    "Swiping Down",  # 15
    "Swiping Left",  # 16
    "Swiping Right",  # 17
    "Swiping Up",  # 18
    "Thumb Down",  # 19
    "Thumb Up",  # 20
    "Turning Hand Clockwise",  # 21
    "Turning Hand Counterclockwise",  # 22
    "Zooming In With Full Hand",  # 23
    "Zooming In With Two Fingers",  # 24
    "Zooming Out With Full Hand",  # 25
    "Zooming Out With Two Fingers"  # 26
]


def process_output(idx_, history):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = 20  # max history buffer

    # mask out illegal action
    if idx_ in [7, 8, 21, 22, 3]:
        idx_ = history[-1]

    # use only single no action class
    if idx_ == 0:
        idx_ = 2
    
    # history smoothing
    if idx_ != history[-1]:
        if not (history[-1] == history[-2]): #  and history[-2] == history[-3]):
            idx_ = history[-1]
    

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history

imgs = []


def video2img():
    cmd = 'ffmpeg  -i \"{}\" -threads 1 -vf scale=-1:331 -q:v 0 \"{}/img_%05d.jpg\"'.format(
          'rtsp://192.168.88.110:1240/jester.264', './image')
    subprocess.call(cmd, shell=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def readimg():
    global imgs
    while True:
        for filename in os.listdir(r"./image"):
            if filename not in imgs:
                imgs.append(filename)


def crop_image(re_img, new_height, new_width):
    re_img = Image.fromarray(np.uint8(re_img))
    width, height = re_img.size
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    crop_im = re_img.crop((left, top, right, bottom))
    crop_im = np.asarray(crop_im)
    return crop_im


def main():
    index = 0
    time.sleep(10)
    buffer = [
        sdk.Tensor(np.zeros([1, 3, 56, 56], dtype=np.float32)),
        sdk.Tensor(np.zeros([1, 4, 28, 28], dtype=np.float32)),
        sdk.Tensor(np.zeros([1, 4, 28, 28], dtype=np.float32)),
        sdk.Tensor(np.zeros([1, 8, 14, 14], dtype=np.float32)),
        sdk.Tensor(np.zeros([1, 8, 14, 14], dtype=np.float32)),
        sdk.Tensor(np.zeros([1, 8, 14, 14], dtype=np.float32)),
        sdk.Tensor(np.zeros([1, 12, 14, 14], dtype=np.float32)),
        sdk.Tensor(np.zeros([1, 12, 14, 14], dtype=np.float32)),
        sdk.Tensor(np.zeros([1, 20, 7, 7], dtype=np.float32)),
        sdk.Tensor(np.zeros([1, 20, 7, 7], dtype=np.float32))]
    for t in buffer:
        t.to_device(DEVICE_ID)
    md = sdk.model(FILE_PATH, DEVICE_ID) 
    i_frame = -2
    history = [2]
    history_logit = []
    history_timing = []
    while True:
        i_frame += 2
        time.sleep(0.2)        
        filename = imgs[i_frame]
        img = Image.open("./image/" + filename).convert('RGB')
        if img.width > img.height:
            frame_pil = img.resize((round(256 * img.width / img.height), 256))
        else:
            frame_pil = img.resize((256, round(256 * img.height / img.width)))
        image = crop_image(frame_pil, 224, 224).transpose(2, 0, 1)
        img_tran = [0, 0, 0]
        for i in range(3):
            img_tran[0] = (image[0] / 255-0.485) / 0.229
            img_tran[1] = (image[1] / 255-0.456) / 0.224
            img_tran[2] = (image[2] / 255-0.406) / 0.225
        img_tran = np.array(img_tran).astype(np.float32)
        img_tran = sdk.Tensor(img_tran)
        img_tran.to_device(DEVICE_ID)
        inputs = [img_tran, ] + buffer
        outputs = md.infer(inputs)
        buffer = outputs[1:]
        outputs[0].to_host()
        out = outputs[0]  
        feat = np.array(out)
        idx_ = np.argmax(feat, axis=1)[0]
        history_logit.append(feat)
        history_logit = history_logit[-12:]
        avg_logit = sum(history_logit)
        idx_ = np.argmax(avg_logit, axis=1)[0]            
        idx, history = process_output(idx_, history)

        print(f"{index} {catigories[idx]}")
        index += 1
s1 = threading.Thread(target=video2img)
s2 = threading.Thread(target=readimg)
if __name__ == '__main__':
    s1.start()
    s2.start()
    main()