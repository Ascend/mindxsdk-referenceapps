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

import argparse
import json
import logging
import os
import stat
import shutil
import time
import xml.dom.minidom
import xml.etree.ElementTree as ET

import cv2
import mmcv
import numpy as np
from PIL import Image

from draw_predict import draw_label
from infer import SdkApi
import config as cfg
from eval_by_sdk import get_eval_result
from postprocess import post_process


def parser_args():
    parser = argparse.ArgumentParser(description="FasterRcnn inference")

    parser.add_argument("--img_path",
                        type=str,
                        required=False,
                        default="../data/test/crop/",
                        help="image directory.")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="../pipeline/fasterrcnn_ms_dvpp.pipeline",
        help="image file path. The default is 'config/maskrcnn_ms.pipeline'. ")
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default="dvpp",
        help=
        "rgb: high-precision, dvpp: high performance. The default is 'dvpp'.")
    parser.add_argument(
        "--infer_mode",
        type=str,
        required=False,
        default="infer",
        help=
        "infer:only infer, eval: accuracy evaluation. The default is 'infer'.")
    parser.add_argument(
        "--infer_result_dir",
        type=str,
        required=False,
        default="../data/test/infer_result",
        help=
        "cache dir of inference result. The default is '../data/test/infer_result'.")
    parser.add_argument("--ann_file",
                        type=str,
                        required=False,
                        help="eval ann_file.")

    arg = parser.parse_args()
    return arg


def get_img_metas(file_name):
    img = Image.open(file_name)
    img_size = img.size

    org_width, org_height = img_size
    resize_ratio = cfg.MODEL_WIDTH / org_width
    if resize_ratio > cfg.MODEL_HEIGHT / org_height:
        resize_ratio = cfg.MODEL_HEIGHT / org_height

    img_metas = np.array([img_size[1], img_size[0]] +
                         [resize_ratio, resize_ratio])
    return img_metas


def process_img(img_file):
    img = cv2.imread(img_file)
    model_img = mmcv.imrescale(img, (cfg.MODEL_WIDTH, cfg.MODEL_HEIGHT))
    if model_img.shape[0] > cfg.MODEL_HEIGHT:
        model_img = mmcv.imrescale(model_img,
                                   (cfg.MODEL_HEIGHT, cfg.MODEL_HEIGHT))
    pad_img = np.zeros(
        (cfg.MODEL_HEIGHT, cfg.MODEL_WIDTH, 3)).astype(model_img.dtype)
    pad_img[0:model_img.shape[0], 0:model_img.shape[1], :] = model_img
    pad_img.astype(np.float16)
    return pad_img


def crop_on_slide(cut_path, crop_path, stride):
    if not os.path.exists(crop_path):
        os.mkdir(crop_path)
    else:
        remove_list = os.listdir(crop_path)
        for filename in remove_list:
            os.remove(os.path.join(crop_path, filename))

    output_shape = 600
    imgs = os.listdir(cut_path)

    for img in imgs:
        if img.split('.')[1] != "jpg" and img.split('.')[1] != "JPG":
            raise ValueError("The file {} is not jpg or JPG image!".format(img))
        origin_image = cv2.imread(os.path.join(cut_path, img))
        height = origin_image.shape[0]
        width = origin_image.shape[1]
        x = 0
        newheight = output_shape
        newwidth = output_shape

        while x < width:
            y = 0
            if x + newwidth <= width:
                while y < height:
                    if y + newheight <= height:
                        hmin = y
                        hmax = y + newheight
                        wmin = x
                        wmax = x + newwidth
                    else:
                        hmin = height - newheight
                        hmax = height
                        wmin = x
                        wmax = x + newwidth
                        y = height  # test

                    crop_img = os.path.join(crop_path, (
                            img.split('.')[0] + '_' + str(wmax) + '_' + str(hmax) + '_' + str(output_shape) + '.jpg'))
                    cv2.imwrite(crop_img, origin_image[hmin: hmax, wmin: wmax])
                    y = y + stride
                    if y + output_shape == height:
                        y = height
            else:
                while y < height:
                    if y + newheight <= height:
                        hmin = y
                        hmax = y + newheight
                        wmin = width - newwidth
                        wmax = width
                    else:
                        hmin = height - newheight
                        hmax = height
                        wmin = width - newwidth
                        wmax = width
                        y = height  # test

                    crop_img = os.path.join(crop_path, (
                            img.split('.')[0] + '_' + str(wmax) + '_' + str(hmax) + '_' + str(
                        output_shape) + '.jpg'))
                    cv2.imwrite(crop_img, origin_image[hmin: hmax, wmin: wmax])
                    y = y + stride
                x = width
            x = x + stride
            if x + output_shape == width:
                x = width


def image_inference(pipeline_path, s_name, img_dir, result_dir,
                    rp_last, model_type):
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_data_plugin_id = 0
    img_metas_plugin_id = 1
    logging.info("\nBegin to inference for {}.\n\n".format(img_dir))

    file_list = os.listdir(img_dir)
    total_len = len(file_list)
    if total_len == 0:
        logging.info("ERROR\nThe input directory is EMPTY!\nPlease place the picture in '../data/test/cut'!")
    for img_id, file_name in enumerate(file_list):
        if not file_name.lower().endswith((".jpg", "jpeg")):
            continue
        file_path = os.path.join(img_dir, file_name)
        save_path = os.path.join(result_dir,
                                 f"{os.path.splitext(file_name)[0]}.json")
        if not rp_last and os.path.exists(save_path):
            logging.info("The infer result json({}) has existed, will be skip.".format(save_path))
            continue

        try:
            if model_type == 'dvpp':
                with open(file_path, "rb") as fp:
                    data = fp.read()
                sdk_api.send_data_input(s_name, img_data_plugin_id, data)
            else:
                img_np = process_img(file_path)
                sdk_api.send_img_input(s_name,
                                       img_data_plugin_id, "appsrc0",
                                       img_np.tobytes(), img_np.shape)

            # set image data
            img_metas = get_img_metas(file_path).astype(np.float32)
            sdk_api.send_tensor_input(s_name, img_metas_plugin_id,
                                      "appsrc1", img_metas.tobytes(), [1, 4],
                                      cfg.TENSOR_DTYPE_FLOAT32)

            start_time = time.time()
            result = sdk_api.get_result(s_name)
            end_time = time.time() - start_time

            if os.path.exists(save_path):
                os.remove(save_path)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open((save_path), flags, modes), 'w') as fp:
                fp.write(json.dumps(result))
            logging.info(
                "End-2end inference, file_name: {}, {}/{}, elapsed_time: {}.\n".format(file_path, img_id + 1, total_len,
                                                                                       end_time))

            draw_label(save_path, file_path, result_dir)
        except Exception as ex:
            logging.exception("Unknown error, msg:{}.".format(ex))
    post_process()


if __name__ == "__main__":
    args = parser_args()

    REPLACE_LAST = True
    STREAM_NAME = cfg.STREAM_NAME.encode("utf-8")
    CUT_PATH = "../data/test/cut/"
    CROP_IMG_PATH = "../data/test/crop/"
    STRIDE = 450
    crop_on_slide(CUT_PATH, CROP_IMG_PATH, STRIDE)
    image_inference(args.pipeline_path, STREAM_NAME, args.img_path,
                    args.infer_result_dir, REPLACE_LAST, args.model_type)
    if args.infer_mode == "eval":
        logging.info("Infer end.\nBegin to eval...")
        get_eval_result(args.ann_file, args.infer_result_dir)
