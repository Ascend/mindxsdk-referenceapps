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
from utils import polyiou
import copy
import cv2

## the IoU thresh for nms when merge image
nms_thresh = 0.3

def py_cpu_nms_poly(dets, thresh):
    """
    任意四点poly nms.取出nms后的边框的索引
    @param dets: shape(detection_num, [poly, confidence1]) 原始图像中的检测出的目标数量
    @param thresh:
    @return:
            keep: 经nms后的目标边框的索引
    """  
    scores = dets[:, 8]
    polys = []
    areas = []
    for i in range(len(dets)):
        tm_polygon = [dets[i][0], dets[i][1],
                      dets[i][2], dets[i][3],
                      dets[i][4], dets[i][5],
                      dets[i][6], dets[i][7]]
        polys.append(tm_polygon)

    # argsort将元素小到大排列 返回索引值 [::-1]即从后向前取元素
    order = scores.argsort()[::-1]  # 取出元素的索引值 顺序为从大到小
    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]  # 取出当前剩余置信度最大的目标边框的索引
        keep.append(i)
        for j in range(order.size - 1):  # 求出置信度最大poly与其他所有poly的IoU
            iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)
        inds = np.where(ovr <= thresh)[0]  # 找出iou小于阈值的索引
        order = order[inds + 1]
    return keep

def nmsbynamedict(nameboxdict, nameboxdict_classname, nms, thresh):
    """
    对namedict中的目标信息进行nms.不改变输入的数据形式
    @param nameboxdict: eg:{
                           'P706':[[poly1, confidence1], ..., 
                                   [poly9, confidence9]],
                           ...
                           'P700':[[poly1, confidence1], ..., 
                                   [poly9, confidence9]]
                            }
    @param nameboxdict_classname: eg:{
                           'P706':[[poly1, confidence1,'classname'], ..., 
                                   [poly9, confidence9, 'classname']],
                           ...
                           'P700':[[poly1, confidence1, 'classname'], ..., 
                                   [poly9, confidence9, 'classname']]
                            }
    @param nms:
    @param thresh: nms阈值, IoU阈值
    @return:
            nameboxnmsdict: eg:{
                                'P706':[[poly1, confidence1, 'classname'], ..., 
                                        [poly_nms, confidence9, 'classname']],
                                 ...
                                'P700':[[poly1, confidence1, 'classname'], ..., 
                                        [poly_nms, confidence9, 'classname']]
                               }
    """
    # 初始化字典
    nameboxnmsdict = {x: [] for x in nameboxdict}  # eg: nameboxnmsdict={'P0770': [], 'P1888': []}
    for imgname in nameboxdict:  # 提取nameboxdict中的key eg:P0770   P1888
        keep = nms(np.array(nameboxdict[imgname]), thresh)  # rotated_nms索引值列表
        outdets = []
        for index in keep:
            outdets.append(nameboxdict_classname[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict

def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly)/2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly

def GetFileFromThisRootDir(dir,ext = None):
    allfiles = []
    needExtFilter = (ext != None)
    for root,dirs,files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles     

def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])

def mergebase(srcpath, dstpath, nms):
    """
    将源路径中所有的txt目标信息,经nms后存入目标路径中的同名txt
    @param srcpath: 合并前信息保存的txt源路径
    @param dstpath: 合并后信息保存的txt目标路径
    @param nms: NMS函数
    """
    # srcpath文件夹下的所有文件相对路径 eg:['example_split/../P0001.txt', ..., '?.txt']
    filelist = GetFileFromThisRootDir(srcpath)  
    for fullname in filelist:  # 'example_split/../P0001.txt'
        name = custombasename(fullname)  # 只留下文件名 eg:P0001
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
                # splitline = [待merge图片名(该目标所处图片名称), confidence, x1, y1, x2, y2, x3, y3, x4, y4]
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
                origpoly = poly2origpoly(poly, x, y, rate)  # 将目标位置信息resize 恢复成原图的poly坐标
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
            nameboxnmsdict = nmsbynamedict(nameboxdict, nameboxdict_classname, nms, nms_thresh)  
            with open(dstname, 'w') as f_out:
                for imgname in nameboxnmsdict:
                    for det in nameboxnmsdict[imgname]:  # 取出对应图片的nms后的目标信息
                        #print('det:', det)
                        confidence = det[-2]
                        bbox = det[0:-2]
                        outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox)) + ' ' + det[-1]
                        #print('outline:', outline)
                        f_out.write(outline + '\n')
            print(fullname + " merge down!")

def mergebypoly(srcpath, dstpath):
    """
    srcpath: result files before merge and nms.txt的信息格式为:[P0770__1__0___0 confidence poly]
    dstpath: result files after merge and nms.保存的txt信息格式为:[P0770 confidence poly]
    """
    mergebase(srcpath,
              dstpath,
              py_cpu_nms_poly)

def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles

def draw_DOTA_image(imgsrcpath, imglabelspath, dstpath, extractclassname, thickness):
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
    # colors = [[178, 63, 143], [25, 184, 176], [238, 152, 129],....,[235, 137, 120]]随机设置RGB颜色
    np.random.seed(666)
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(extractclassname))]
    filelist = GetFileFromThisRootDir(imglabelspath)  # fileist=['/.../P0005.txt', ..., /.../P000?.txt]
    for fullname in filelist:  # fullname='/.../P000?.txt'
        objects = []
        with open(fullname, 'r') as f_in:  # 打开merge后/原始的DOTA图像的gt.txt
            lines = f_in.readlines()  # 读取txt中所有行,每行作为一个元素存于list中
            splitlines = [x.strip().split(' ') for x in lines]  # 再次分割list中的每行元素 shape:n行 * m个元素
            if len(splitlines[0]) == 1:  # 首行为"imagesource:GoogleEarth",说明为DOTA原始labels
                # DOTA labels:[polys classname 1/0]
                del splitlines[0]
                del splitlines[0]   # 删除前两个无用信息
                objects = [x[0:-2] for x in splitlines]
                classnames = [x[-2] for x in splitlines]
            else:
                # P0003 0.911 660.0 309.0 639.0 209.0 661.0 204.0 682.0 304.0 large-vehicle
                confidences = [float(x[1]) for x in splitlines]
                # print(confidences)
                objects = [x[2:-1] for x in splitlines]
                classnames = [x[-1] for x in splitlines]
                # print(classnames)

        '''
        objects[i] = str[poly, classname]
        '''
        name = os.path.splitext(os.path.basename(fullname))[0]  # name='P000?'
        img_fullname = os.path.join(imgsrcpath, name + '.jpg')
        img_savename = os.path.join(dstpath, name + '_.jpg')
        img = cv2.imread(img_fullname)  # 读取图像像素

        tl = 1
        tf = tl-1 if (tl-1>1) else 1

        for i, obj in enumerate(objects):
            # obj = [poly ,'classname']
            classname = classnames[i]
            # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
            poly = np.array(list(map(float, obj)))
            poly = poly.reshape(4, 2)  # 返回rect对应的四个点的值 normalized
            poly = np.int0(poly)

            # 画出来
            cv2.drawContours(image=img,
                             contours=[poly],
                             contourIdx=-1,
                             color=colors[int(extractclassname.index(classname))],
                             thickness=thickness)

            label = '%s %.2f' % (classnames[i], confidences[i])
            rec = cv2.minAreaRect(poly)
            c1 = (int(rec[0][0]), int(rec[0][1]))
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, colors[int(extractclassname.index(classname))], -1, cv2.LINE_AA) 
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [225, 255, 255], 
                        thickness=tf, lineType=cv2.LINE_AA)
        cv2.imwrite(img_savename, img)
        
if __name__ == '__main__':
    classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
                  'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                  'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 
                  'harbor', 'swimming-pool', 'helicopter', 'container-crane']

    mergebypoly(r'/home/zhongzhi8/RotatedObjectDetection/testdetection/result_txt/result_before_merge', 
                r'/home/zhongzhi8/RotatedObjectDetection/testdetection/result_txt/result_merged')

    draw_DOTA_image(imgsrcpath=r'/home/zhongzhi8/RotatedObjectDetection/testImage',
                    imglabelspath=r'/home/zhongzhi8/RotatedObjectDetection/testdetection/result_txt/result_merged',
                    dstpath=r'/home/zhongzhi8/RotatedObjectDetection/testdetection/merged_drawed',
                    extractclassname=classnames,
                    thickness=2
                    )