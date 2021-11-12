# --------------------------------------------------------
# dota_evaluation_task1
# Licensed under The MIT License [see LICENSE for details]
# Written by Jian Ding, based on code from Bharath Hariharan
# --------------------------------------------------------

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
import copy
from utils import polyiou

def parse_gt(filename):
    """
    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with  open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]

                # if (len(splitlines) == 9):
                #     object_struct['difficult'] = 0
                # elif (len(splitlines) == 10):
                #     object_struct['difficult'] = int(splitlines[9])
                object_struct['difficult'] = 0
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ 
    ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=False):
    """
    rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    #if not os.path.isdir(cachedir):
     #   os.mkdir(cachedir)
    #cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    #print('imagenames: ', imagenames)
    #if not os.path.isfile(cachefile):
        # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        #print('parse_files name: ', annopath.format(imagename))
        recs[imagename] = parse_gt(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    #print('check confidence: ', confidence)

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    #print('check sorted_scores: ', sorted_scores)
    #print('check sorted_ind: ', sorted_ind)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    #print('check imge_ids: ', image_ids)
    #print('imge_ids len:', len(image_ids))
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]
            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(BBGT_keep[index], bb)
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]
        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.


    print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
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
                           'P706':[[poly1, confidence1], ..., [poly9, confidence9]],
                           ...
                           'P700':[[poly1, confidence1], ..., [poly9, confidence9]]
                            }
    @param nameboxdict_classname: eg:{
                           'P706':[[poly1, confidence1,'classname'], ..., [poly9, confidence9, 'classname']],
                           ...
                           'P700':[[poly1, confidence1, 'classname'], ..., [poly9, confidence9, 'classname']]
                            }
    @param nms:
    @param thresh: nms阈值, IoU阈值
    @return:
            nameboxnmsdict: eg:{
                                'P706':[[poly1, confidence1, 'classname'], ..., [poly_nms, confidence9, 'classname']],
                                 ...
                                'P700':[[poly1, confidence1, 'classname'], ..., [poly_nms, confidence9, 'classname']]
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

def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])

def GetFileFromThisRootDir(dir, ext=None):
  allfiles = []
  needExtFilter = (ext != None)
  for root, dirs, files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles

def mergebase(srcpath, dstpath, nms):
    """
    将源路径中所有的txt目标信息,经nms后存入目标路径中的同名txt
    @param srcpath: 合并前信息保存的txt源路径
    @param dstpath: 合并后信息保存的txt目标路径
    @param nms: NMS函数
    """
    filelist = GetFileFromThisRootDir(srcpath)  # srcpath文件夹下的所有文件相对路径 eg:['example_split/../P0001.txt', ..., '?.txt']
    for fullname in filelist:  # 'example_split/../P0001.txt'
        name = custombasename(fullname)  # 只留下文件名 eg:P0001
        dstname = os.path.join(dstpath, name + '.txt')  # eg: example_merge/..P0001.txt
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        with open(fullname, 'r') as f_in:
            nameboxdict = {}
            nameboxdict_classname = {}
            lines = f_in.readlines()  # 读取txt中所有行,每行作为一个元素存于list中
            splitlines = [x.strip().split(' ') for x in lines]  # 再次分割list中的每行元素 shape:n行 * m个元素
            for splitline in splitlines:  # splitline:每行中的m个元素
                # splitline = [待merge图片名(该目标所处图片名称), confidence, x1, y1, x2, y2, x3, y3, x4, y4, classname]
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
                det.append(confidence)  # [poly, 'confidence']
                det = list(map(float, det))  # [poly, confidence]

                det_classname = copy.deepcopy(det)
                det_classname.append(classname)  # [poly, 'confidence','classname']
                if (oriname not in nameboxdict):
                    nameboxdict[oriname] = []   # 弄个元组,汇集原图目标信息 eg: 'P706':[[poly1, confidence1], ..., ]
                    nameboxdict_classname[oriname] = []   # 弄个元组,汇集原图目标信息 eg: 'P706':[[poly1, confidence1,'classname'], ..., ]
                nameboxdict[oriname].append(det)
                nameboxdict_classname[oriname].append(det_classname)

            nameboxnmsdict = nmsbynamedict(nameboxdict, nameboxdict_classname, nms, thresh=0.3)  # 对nameboxdict元组进行nms
            with open(dstname, 'w') as f_out:
                for imgname in nameboxnmsdict:  # 'P706'
                    for det in nameboxnmsdict[imgname]:  # 取出对应图片的nms后的目标信息
                        # det:[poly1, confidence1, 'classname']
                        confidence = det[-2]
                        bbox = det[0:-2]
                        outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox)) + ' ' + det[-1]
                        f_out.write(outline + '\n')
            print(name, "merge down!")

def mergebypoly(srcpath, dstpath):
    """
    @param srcpath: result files before merge and nms.txt的信息格式为:[P0770__1__0___0 confidence poly 'classname']
    @param dstpath: result files after merge and nms.保存的txt信息格式为:[P0770 confidence poly 'classname']
    """
    # srcpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_test_results'
    # dstpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/testtime'

    mergebase(srcpath,
              dstpath,
              py_cpu_nms_poly)

def image2txt(srcpath, dstpath):
    """
    将srcpath文件夹下的所有子文件名称打印到namefile.txt中
    @param srcpath: imageset
    @param dstpath: imgnamefile.txt的存放路径
    """
    filelist = GetFileFromThisRootDir(srcpath)  # srcpath文件夹下的所有文件相对路径 eg:['example_split/../P0001.txt', ..., '?.txt']
    for fullname in filelist:  # 'example_split/../P0001.txt'
        name = custombasename(fullname)  # 只留下文件名 eg:P0001
        dstname = os.path.join(dstpath, 'imgnamefile.txt')  # eg: result/imgnamefile.txt
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        with open(dstname, 'a') as f:
            f.writelines(name + '\n')

def evaluation_trans(srcpath, dstpath):
    """
    将srcpath文件夹中的所有txt中的目标提取出来,按照目标类别分别存入 Task1_类别名.txt中:
            txt中的内容格式:  目标所属原始图片名称 置信度 poly
    @param srcpath: 存放图片的目标检测结果(文件夹,内含多个txt)
                    txt中的内容格式: 目标所属图片名称 置信度 poly 'classname'
    @param dstpath: 存放图片的目标检测结果(文件夹, 内含多个Task1_类别名.txt )
                    txt中的内容格式:  目标所属原始图片名称 置信度 poly
    """
    filelist = GetFileFromThisRootDir(srcpath)  # srcpath文件夹下的所有文件相对路径 eg:['result_merged/P0001.txt', ..., '?.txt']
    for fullname in filelist:  # 'result_merged/P0001.txt'
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        with open(fullname, 'r') as f_in:
            lines = f_in.readlines()  # 读取txt中所有行,每行作为一个元素存于list中
            splitlines = [x.strip().split(' ') for x in lines]  # 再次分割list中的每行元素 shape:n行 * m个元素
            for splitline in splitlines:  # splitline:每行中的m个元素
                # splitline = [目标所属图片名称, confidence, x1, y1, x2, y2, x3, y3, x4, y4, 'classname']
                classname = splitline[-1]  # 每行的最后一个元素 是被分割的图片的种类名
                dstname = os.path.join(dstpath, 'Task1_' + classname + '.txt')  # eg: result/Task1_plane.txt
                lines_ = ' '.join(list(splitline[:-1]))
                with open(dstname, 'a') as f:
                    f.writelines(lines_ + '\n')


def evaluation(detoutput, imageset, annopath, classnames):
    """
    评估程序
    @param detoutput: detect.py函数的结果存放输出路径
    @param imageset: # val DOTA原图数据集图像路径
    @param annopath: 'r/.../{:s}.txt'  原始val测试集的DOTAlabels路径
    @param classnames: 测试集中的目标种类
    """
    result_before_merge_path = str(detoutput + '/result_txt/result_before_merge')
    result_merged_path = str(detoutput + '/result_txt/result_merged')
    # result_merged_path = str(detoutput + '/result_txt/result_before_merge')
    result_classname_path = str(detoutput + '/result_txt/result_classname')
    imageset_name_file_path = str(detoutput + '/result_txt')

    # see demo for example

    
    mergebypoly(
        result_before_merge_path,
        result_merged_path
    )
    print('检测结果已merge')
    evaluation_trans(
        result_merged_path,
        result_classname_path
    )
    print('检测结果已按照类别分类')
    image2txt(
        imageset,  # val原图数据集路径
        imageset_name_file_path
              )
    print('校验数据集名称文件已生成')

    detpath = str(result_classname_path + '/Task1_{:s}.txt')  # 'r/.../Task1_{:s}.txt'  存放各类别结果文件txt的路径
    annopath = annopath
    imagesetfile = str(imageset_name_file_path + '/imgnamefile.txt')  # 'r/.../imgnamefile.txt'  测试集图片名称txt

    # detpath = r'PATH_TO_BE_CONFIGURED/Task1_{:s}.txt'
    # annopath = r'PATH_TO_BE_CONFIGURED/{:s}.txt' # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    # imagesetfile = r'PATH_TO_BE_CONFIGURED/valset.txt'

    # For DOTA-v1.5
    #classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #            'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']
    # For DOTA-v1.0
    # classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #             'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', ']
    classaps = []
    map = 0
    skippedClassCount = 0
    for classname in classnames:
        print('classname:', classname)
        detfile = detpath.format(classname)
        if not (os.path.exists(detfile)):
            skippedClassCount += 1
            print('This class is not be detected in your dataset: {:s}'.format(classname))
            continue
        rec, prec, ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=True)
        map = map + ap
        print('ap: ', ap)
        classaps.append(ap)

    map = map / (len(classnames) - skippedClassCount)
    print('map:', map)
    classaps = 100 * np.array(classaps)
    print('classaps: ', classaps)



if __name__ == '__main__':
    
    classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
                  'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                  'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 
                  'harbor', 'swimming-pool', 'helicopter', 'container-crane']

    #
    evaluation(
        detoutput='/home/zhongzhi8/RotatedObjectDetection/detection_plugin',
        imageset=r'/home/zhongzhi8/RotatedObjectDetection/dataSet/images',
        annopath=r'/home/zhongzhi8/RotatedObjectDetection/dataSet/labelTxt/{:s}.txt',
        classnames=classnames
    )

