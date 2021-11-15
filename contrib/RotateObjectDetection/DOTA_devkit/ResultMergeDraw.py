# -*- coding: utf-8 -*-
# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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

"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""

import os
import numpy as np
import re
from utils import general_utils
import copy
import cv2
import argparse

## the IoU thresh for nms when merge image
nms_thresh = 0.3

def mergebase(srcpath, dstpath, nms):
    """
    将源路径中所有的txt目标信息,经nms后存入目标路径中的同名txt
    @param srcpath: 合并前信息保存的txt源路径
    @param dstpath: 合并后信息保存的txt目标路径
    @param nms: NMS函数
    """
    # srcpath文件夹下的所有文件相对路径 eg:['example_split/../P0001.txt', ..., '?.txt']
    filelist = general_utils.getfilefromthisrootdir(srcpath)  
    for fullname in filelist:  # 'example_split/../P0001.txt'
        name = general_utils.custombasename(fullname)  # 只留下文件名 eg:P0001
        dstname = os.path.join(dstpath, name + '.txt')  # eg: example_merge/..P0001.txt
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        with open(fullname, 'r') as f_in:
            nameboxdict = {}
            nameboxdict_classname = {}
            # 读取txt中所有行,每行作为一个元素存于list中
            lines = f_in.readlines()  
             # 再次分割list中的每行元素 shape:n行 * m个元素
            splitlines = [x.strip().split(' ') for x in lines] 
            for splitline in splitlines:  # splitline:每行中的m个元素
                subname = splitline[0]  # 每行的第一个元素 是被分割的图片的图片名 eg:P0706__1__0___0
                splitname = subname.split('__')  # 分割待merge的图像的名称 eg:['P0706','1','0','_0']
                oriname = splitname[0]  # 获得待merge图像的原图像名称 eg:P706
                pattern1 = re.compile(r'__\d+___\d+')  # 预先编译好r'__\d+___\d+' 提高重复使用效率 \d表示数字

                x_y = re.findall(pattern1, subname)  # 匹配subname中的字符串 eg: x_y=['__0___0']
                x_y_2 = re.findall(r'\d+', x_y[0])  # 匹配subname中的字符串 eg: x_y_2= ['0','0']
                x, y = int(x_y_2[0]), int(x_y_2[1])  # 找到当前subname图片在原图中的分割位置

                pattern2 = re.compile(r'__([\d+\.]+)__\d+___')  # \.表示一切字符

                rate = re.findall(pattern2, subname)[0]  # 找到该subname分割图片时的分割rate (resize rate before cut)

                confidence = splitline[1]
                classname = splitline[-1]
                poly = list(map(float, splitline[2:10]))  # 每个元素映射为浮点数 再放入列表中
                origpoly = general_utils.poly2origpoly(poly, x, y, rate)  # 将目标位置信息resize 恢复成原图的poly坐标
                det = origpoly  # shape(8)
                det.append(confidence)  # [poly, confidence]
                det = list(map(float, det))
                det_classname = copy.deepcopy(det)
                det_classname.append(classname)  # [poly, 'confidence','classname']
                if (oriname not in nameboxdict):
                     # 弄个元组,汇集原图目标信息 eg: 'P706':[[poly1, confidence1], ..., ]
                    nameboxdict[oriname] = []  
                    # 弄个元组,汇集原图目标信息 eg: 'P706':[[poly1, confidence1,'classname'], ..., ]
                    nameboxdict_classname[oriname] = []   
                nameboxdict[oriname].append(det)
                nameboxdict_classname[oriname].append(det_classname)
            # 对nameboxdict元组进行nms
            nameboxnmsdict = general_utils.nmsbynamedict(nameboxdict, nameboxdict_classname, nms, nms_thresh)  
            with open(dstname, 'w') as f_out:
                for imgname in nameboxnmsdict:
                    for det in nameboxnmsdict[imgname]:  # 取出对应图片的nms后的目标信息
                        confidence = det[-2]
                        bbox = det[0:-2]
                        outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox)) + ' ' + det[-1]
                        f_out.write(outline + '\n')
            print(name + " merge down!")

def mergebypoly(srcpath, dstpath):
    """
    srcpath: result files before merge and nms.txt的信息格式为:[P0770__1__0___0 confidence poly]
    dstpath: result files after merge and nms.保存的txt信息格式为:[P0770 confidence poly]
    """
    mergebase(srcpath,
              dstpath,
              general_utils.py_cpu_nms_poly)

def draw_dota_image(imgsrcpath, imglabelspath, dstpath, extractclassname, thickness, labels):
    """
    绘制工具merge后的目标/DOTA GT图像
        @param imgsrcpath: merged后的图像路径(原始图像路径)
        @param imglabelspath: merged后的labels路径
        @param dstpath: 目标绘制之后的保存路径
        @param extractclassname: the category you selected
    """
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
    # 设置画框的颜色    
    np.random.seed(666)
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(extractclassname))]
    filelist = general_utils.getfilefromthisrootdir(imglabelspath)  
    for fullname in filelist: 
        objects = []
        with open(fullname, 'r') as f_in:  # 打开merge后/原始的DOTA图像的gt.txt
            lines = f_in.readlines()  # 读取txt中所有行,每行作为一个元素存于list中
            splitlines = [x.strip().split(' ') for x in lines]  # 再次分割list中的每行元素 shape:n行 * m个元素
            if len(splitlines[0]) == 1:  # 首行为"imagesource:GoogleEarth",说明为DOTA原始labels
                del splitlines[0]
                del splitlines[0]   # 删除前两个无用信息
                objects = [x[0:-2] for x in splitlines]
                class_names = [x[-2] for x in splitlines]
            else:
                confidences = [float(x[1]) for x in splitlines]
                objects = [x[2:-1] for x in splitlines]
                class_names = [x[-1] for x in splitlines]

        '''
        objects[i] = str[poly, classname]
        '''
        name = os.path.splitext(os.path.basename(fullname))[0]  # name='P000?'
        img_fullname = os.path.join(imgsrcpath, name + '.jpg')
        img_savename = os.path.join(dstpath, name + '_.jpg')
        img = cv2.imread(img_fullname)  # 读取图像像素

        tl = 1
        tf = tl - 1 if (tl - 1 > 1) else 1

        for i, obj in enumerate(objects):
            classname = class_names[i]
            poly = np.array(list(map(float, obj)))
            poly = poly.reshape(4, 2)  # 返回rect对应的四个点的值 normalized
            poly = np.int0(poly)

            # 画出来
            cv2.drawContours(image=img,
                             contours=[poly],
                             contourIdx=-1,
                             color=colors[int(extractclassname.index(classname))],
                             thickness=thickness)
            if labels:
                label = '%s %.2f' % (class_names[i], confidences[i])
            else:
                label = '%s' % int(extractclassname.index(classname))
            rec = cv2.minAreaRect(poly)
            c1 = (int(rec[0][0]), int(rec[0][1]))
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, colors[int(extractclassname.index(classname))], -1, cv2.LINE_AA) 
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [225, 255, 255], 
                        thickness=tf, lineType=cv2.LINE_AA)
        cv2.imwrite(img_savename, img)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_print', action='store_true', default=False, help='whether to print labels')    
    opt = parser.parse_args()
    labels = opt.labels_print

    classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
                  'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                  'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 
                  'harbor', 'swimming-pool', 'helicopter', 'container-crane']

    mergebypoly(srcpath=r'../detection/result_txt/result_before_merge', 
                dstpath=r'../detection/result_txt/result_merged')

    draw_dota_image(imgsrcpath=r'../image',
                    imglabelspath=r'../detection/result_txt/result_merged',
                    dstpath=r'../detection/merged_drawed',
                    extractclassname=classnames,
                    thickness=2,
                    labels=labels
                    )