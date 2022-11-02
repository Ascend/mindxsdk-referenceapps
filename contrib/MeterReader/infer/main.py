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
import cv2


cur_path = os.path.abspath(os.path.dirname(__file__))

father_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def get_args():
    argv = sys.argv[1:]
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    print('输入的文件为：', inputfile)
    print('输出的文件为：', outputfile)
    return inputfile,outputfile



if __name__ == '__main__':


    # python main.py --ifile /home/wangyi4/jiang/all/images/test.jpg --ofile /home/wangyi4/jiang/all/images/det_res.jpg
    inputfile,outputfile = get_args()
    # det
    # inputfile = "/home/wangyi4/tmp/221021_xhr/images/det_res.jpg"
    outputdir = os.path.join(father_path, 'images', 'det_res/').replace('\\', '/')

    # outputdir = "/home/wangyi4/tmp/221021_xhr/images/det_rect/" # 写定，临时文件夹
    det_result = os.popen(f'python det_test.py --ifile {inputfile} --odir {outputdir}')  
    det_res = det_result.read() 

    det_res_img_len = 0
    det_res_img_file = []
    det_res_img_index = []
    for line in det_res.splitlines(): 
        if (line.startswith("res_img")):
            temp = line.split(" ")
            det_res_img_len = int(temp[1])
            for i in range(det_res_img_len):
                det_res_img_file.append(temp[2+i])
        elif line.startswith("det_xyxy"):
            temp = line.split(":")[1]
            print(json.loads(temp))
            det_res_img_index.append(json.loads(temp))

        print(line)  

    
    # print("------det-----")
    # print(det_res_img_len)
    # print(det_res_img_file)

    # seg
    seg_Ans = []
    for i in range(det_res_img_len):
        # seg_result = os.popen(f"python seg.py --ifile {det_res_img_file[i]} --ofile /home/wangyi4/tmp/221021_xhr/images/det_rect/")
        seg_result = os.popen(f"python seg.py --ifile {det_res_img_file[i]} --ofile {outputdir}")
        seg_res = seg_result.read() 
        for line in seg_res.splitlines(): 
            if (line.startswith("seg_ans")):
                # print(line)
                temp = line.split(" ")
                # print(temp)
                seg_Ans.append(float(temp[1]))
            # print(line)
        os.remove(det_res_img_file[i])
    print("------seg-----")
    print(seg_Ans)

    print("------All-----")
    for i in range(det_res_img_len):
        print(f"{det_res_img_file[i]}   {seg_Ans[i]}")
            # print(line)
        
    # 处理成为图片
    im0 = cv2.imread(inputfile)
    for i in range(len(det_res_img_index)):
        meter = det_res_img_index[i]
        #加边框
        cv2.rectangle(im0, (int(meter[0]), int(meter[1])), (int(meter[2]), int(meter[3])), (0, 255, 0), 2)
        #加读数
        text = f"meter  {seg_Ans[i]:.3f}"
        im0 = cv2.putText(im0, text, (int(meter[0]+3), int(meter[1])+30), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA, False)

    cv2.imwrite(outputfile, im0)
