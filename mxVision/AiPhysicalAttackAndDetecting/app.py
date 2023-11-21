#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
Description: Web system.
Author: MindX SDK
Create: 2023
History: NA
"""

import time
from multiprocessing import Process, Queue
import cv2
from flask import Flask, render_template, Response, jsonify
from ultrayolo_attack_detector.detect_attack_by_mxbase import DetectObjWithYolov3OMUltra
from darknet_ob_detector.detect_ob_by_mxbase import DetectObjWithYolov3OM
from camera.camera_process import get_frame
from args import input_args_parse
from mindx.sdk import base


app = Flask(__name__, static_url_path='')
base.mx_init()
found_person = False
found_patch = False
cmu_detector = DetectObjWithYolov3OM()
ultra_detector = DetectObjWithYolov3OMUltra()
my_queue = Queue(maxsize=3)


@app.route('/')
def index():
    return render_template('index.html')


def gen0(my_queue):
    """Video streaming generator function."""
    global found_person
    while True:
        frm = my_queue.get()
        frm_with_bbox, found_person = cmu_detector.detect_image_with_yolov3_om(frm)
        _, frm_encode = cv2.imencode('.jpg', frm_with_bbox)
        frm_bytes = frm_encode.tobytes()
        yield b'Content-Type: image/jpeg\r\n\r\n' + frm_bytes + b'\r\n--frame\r\n'


def gen1(my_queue):
    first = False
    last = time.time()

    global found_patch
    while True:
        frm = my_queue.get()

        frm_with_bbox, found_patch = ultra_detector.detect_image_with_yolov3_om(frm)
        _, frm_encode = cv2.imencode('.jpg', frm_with_bbox)
        frm_bytes = frm_encode.tobytes()
        if found_patch and not first:
            first = True
            last = time.time()
        now = time.time()
        interval = int(now - last)

        if found_patch and first and interval > 3:
            last = int(now)
        yield b'Content-Type: image/jpeg\r\n\r\n' + frm_bytes + b'\r\n--frame\r\n'


@app.route('/video_feed0')
def video_feed0():
    print('video_feed 0')
    return Response(gen0(my_queue), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed1')
def video_feed1():
    print('video_feed 1')
    return Response(gen1(my_queue), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_data')
def get_data():
    alert1 = "警告：监控区域有人进入" if found_person else "没有识别到人"
    alert2 = "警告：监控区域发现对抗攻击" if found_patch else "没有识别到AI攻击"
    cl1 = "red" if found_person else "black"
    cl2 = "red" if found_patch else "black"
    data_dict = {'left': alert1, 'right': alert2, 'cll': cl1, 'clr': cl2}
    return jsonify(data_dict)


if __name__ == "__main__":
    input_args = input_args_parse()
    my_process = Process(target=get_frame, args=(my_queue,))
    my_process.start()
    app.run(host=input_args.host, port=input_args.port)
    my_process.join()
