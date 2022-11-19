#!/usr/bin/env python
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

import os
import sys
import json
import getopt
import shutil
import cv2

cur_path = os.path.abspath(os.path.dirname(__file__))

father_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def error(msg):
    print("ERROR!!! ", msg)
    sys.exit()


def get_args():
    argv = sys.argv[1:]
    input_file = ''
    output_dir = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "odir="])
    except getopt.GetoptError as e:
        error(e)
    opt_ifile_flag = False
    opt_odir_flag = False
    for opt, arg in opts:
        if opt == '-h':
            sys.exit()
        elif opt in ("-i", "--ifile"):
            opt_ifile_flag = True
            input_file = arg
        elif opt in ("-o", "--odir"):
            opt_odir_flag = True
            output_dir = arg
    if (not opt_ifile_flag):
        error("未输入参数--ifile。请参考python main.py --ifile <input_file_path> --odir <output_dir_path>")
    elif (not opt_odir_flag):
        error("未输入参数--odir。请参考python main.py --ifile <input_file_path> --odir <output_dir_path>")

    # 判定是否输入文件名为空
    if (not input_file):
        error("输入文件名为空")
    # 判定是否为cv2可读文件
    test_cv_read(input_file)
    print('输入的文件为：', input_file)

    if (not output_dir):
        error("输出文件目录为空")
    elif (not os.path.isdir(output_dir)):
        error("输出文件目录不存在，请创建。")
    print('输出的文件目录为：', output_dir)
    return [input_file, output_dir]


# 判断改文件是否可被cv2读取，或者为空图像
def test_cv_read(filename):
    try:
        temp_img = cv2.imread(filename, -1)
        if (temp_img is None):
            error(f"cv2读取文件失败 或 图像为空. {filename}")
        return
    except Exception as e:
        print(e)
        error(f"cv2读取文件失败 或 图像为空. {filename}")


if __name__ == '__main__':
    inputfile, outputdir = get_args()

    # det
    outputdir_temp = os.path.join(father_path, 'images', 'det_res/').replace('\\', '/')

    inputfilename = os.path.basename(inputfile)
    dirStr, ext = os.path.splitext(inputfilename)
    outputfile = os.path.join(outputdir, dirStr + "_result" + ".png")

    # 如果yolov5输出的文件夹不存在，则创建。存在则清空再创建。
    if (not os.path.isdir(outputdir_temp)):
        os.makedirs(outputdir_temp)
    else:
        shutil.rmtree(outputdir_temp)
        os.makedirs(outputdir_temp)

    det_result = os.popen(f'python det.py --ifile {inputfile} --odir {outputdir_temp}')
    det_res = det_result.read()

    DET_RES_IMG_LEN = 0
    detResImgFile = []
    detResImgIndex = []
    for line in det_res.splitlines():
        if (line.startswith("res_img")):
            temp = line.split(" ")
            DET_RES_IMG_LEN = int(temp[1])
            for i in range(DET_RES_IMG_LEN):
                detResImgFile.append(temp[2 + i])
            continue
        elif line.startswith("det_xyxy"):
            temp = line.split(":")[1]
            detResImgIndex.append(json.loads(temp))
            continue

        print(line)

    if (len(detResImgFile) == 0):
        error("yolov5未识别到表盘图片")

    if (len(detResImgFile) != len(detResImgIndex)):
        error(f"yolov5识别的表盘图片数量（{len(detResImgFile)}）与框的数量（{len(detResImgIndex)}）不一致。")

    # seg
    seg_Ans = []
    for i in range(DET_RES_IMG_LEN):
        seg_result = os.popen(f"python seg.py --ifile {detResImgFile[i]}")
        seg_res = seg_result.read()
        for line in seg_res.splitlines():
            if (line.startswith("seg_ans")):
                temp = line.split(" ")
                seg_Ans.append(float(temp[1]))
        os.remove(detResImgFile[i])

    # 处理成为图片
    im0 = cv2.imread(inputfile)
    for meter in detResImgIndex:
        # 加边框
        cv2.rectangle(im0, (int(meter[0]), int(meter[1])), (int(meter[2]), int(meter[3])), (0, 255, 0), 2)
        # 加读数
        text = f"meter  {seg_Ans[detResImgIndex.index(meter)]:.3f}"
        im0 = cv2.putText(im0, text, (int(meter[0] + 3), int(meter[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                          2, cv2.LINE_AA, False)

    cv2.imwrite(outputfile, im0)

    # 删去临时文件夹
    shutil.rmtree(outputdir_temp)

    print("Success!")
