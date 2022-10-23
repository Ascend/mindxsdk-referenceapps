# Copyright 2021 Huawei Technologies Co., Ltd
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
"""coco eval for maskrcnn"""
import json

import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json
import shutil
import numpy as np
import cv2 as cv
import tqdm
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import shutil
import xml.etree.ElementTree as ET

_init_value = np.array(0.0)
summary_init = {
    'Precision/mAP': _init_value,
    'Precision/mAP@.50IOU': _init_value,
    'Precision/mAP@.75IOU': _init_value,
    'Precision/mAP (small)': _init_value,
    'Precision/mAP (medium)': _init_value,
    'Precision/mAP (large)': _init_value,
    'Recall/AR@1': _init_value,
    'Recall/AR@10': _init_value,
    'Recall/AR@100': _init_value,
    'Recall/AR@100 (small)': _init_value,
    'Recall/AR@100 (medium)': _init_value,
    'Recall/AR@100 (large)': _init_value,
}


def coco_eval(result_files,
              result_types,
              coco,
              max_dets=(100, 300, 1000),
              single_result=False):
    """coco eval for maskrcnn"""
    anns = json.load(open(result_files['bbox']))
    if not anns:
        return summary_init
    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    for res_type in result_types:
        result_file = result_files[res_type]
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        gt_img_ids = coco.getImgIds()
        det_img_ids = coco_dets.getImgIds()
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)

        tgt_ids = gt_img_ids if not single_result else det_img_ids

        if single_result:
            res_dict = dict()
            for id_i in tgt_ids:
                cocoEval = COCOeval(coco, coco_dets, iou_type)
                if res_type == 'proposal':
                    cocoEval.params.useCats = 0
                    cocoEval.params.maxDets = list(max_dets)

                cocoEval.params.imgIds = [id_i]
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                res_dict.update(
                    {coco.imgs[id_i]['file_name']: cocoEval.stats[1]})

        cocoEval = COCOeval(coco, coco_dets, iou_type)
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)

        cocoEval.params.imgIds = tgt_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        summary_metrics = {
            'Precision/mAP': cocoEval.stats[0],
            'Precision/mAP@.50IOU': cocoEval.stats[1],
            'Precision/mAP@.75IOU': cocoEval.stats[2],
            'Precision/mAP (small)': cocoEval.stats[3],
            'Precision/mAP (medium)': cocoEval.stats[4],
            'Precision/mAP (large)': cocoEval.stats[5],
            'Recall/AR@1': cocoEval.stats[6],
            'Recall/AR@10': cocoEval.stats[7],
            'Recall/AR@100': cocoEval.stats[8],
            'Recall/AR@100 (small)': cocoEval.stats[9],
            'Recall/AR@100 (medium)': cocoEval.stats[10],
            'Recall/AR@100 (large)': cocoEval.stats[11],
        }

    print(json.dumps(summary_metrics, indent=2))

    return summary_metrics


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def det2json(dataset, results):
    """convert det to json"""
    cat_ids = dataset.getCatIds()
    img_ids = dataset.getImgIds()
    json_results = []
    dataset_len = len(img_ids)
    for idx in range(dataset_len):
        img_id = img_ids[idx]
        if idx == len(results): break
        result = results[idx]
        for label, result_label in enumerate(result):
            bboxes = result_label
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = cat_ids[label]
                json_results.append(data)
    return json_results


def results2json(dataset, results, out_file):
    """convert result to json"""
    result_files = dict()
    json_results = det2json(dataset, results)
    result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
    result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
    mmcv.dump(json_results, result_files['bbox'])
    return result_files



def VOC_eval(ann_file, result_json_file, voc_dir):
    savetxt_path = os.path.join(voc_dir, "VOC2017/image_txt")
    valtxt_path = os.path.join(voc_dir, "VOC2017/ImageSets/Main/val.txt")
    coco_to_txt(ann_file, result_json_file, valtxt_path, savetxt_path, cat_id=1)

    txtPath = savetxt_path
    saveTxtPath = os.path.join(voc_dir, "VOC2017/image_huizong_txt")
    removeTxtPath = os.path.join(voc_dir, "VOC2017/image_huizong_txt_nms")
    cut_path = valtxt_path
    hebing_txt(txtPath, saveTxtPath, removeTxtPath, cut_path)

    cut_path = os.path.join(voc_dir, "VOC2017/JPEGImages")
    imagesavePath = os.path.join(voc_dir, "VOC2017/images1")
    txtPath = removeTxtPath
    nms_box(cut_path, imagesavePath, txtPath, thresh=0)

    txtPath = removeTxtPath
    saveTxtPath = os.path.join(voc_dir, "VOC2017/obj_txt_huizong")
    write_huizong(txtPath, saveTxtPath)

    aps = []  # 保存各类ap
    recs = []  # 保存recall
    precs = []  # 保存精度
    # annopath = './VOCdevkit/VOC2017/Annotations/' + '{:s}.xml'  # annotations的路径，{:s}.xml方便后面根据图像名字读取对应的xml文件
    annopath = voc_dir + "/VOC2017/Annotations/" + '{:s}.xml'  # annotations的路径，{:s}.xml方便后面根据图像名字读取对应的xml文件
    imagesetfile = os.path.join(voc_dir, "VOC2017/ImageSets/Main/val.txt")  # 读取图像名字列表文件
    cachedir = os.path.join(voc_dir, "VOC2017/demo")
    filename = os.path.join(voc_dir, "VOC2017/obj_txt_huizong/qikong.txt")

    rec, prec, ap = voc_eval(  # 调用voc_eval.py计算cls类的recall precision ap
        filename, annopath, imagesetfile, "qikong", cachedir, ovthresh=0,
        use_07_metric=False)

    aps += [ap]

    print('AP for {} = {:.4f}'.format('qikong', ap))
    print('recall for {} = {:.4f}'.format('qikong', rec[-1]))
    print('precision for {} = {:.4f}'.format('qikong', prec[-1]))
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')


def coco_to_txt(annotation_file, res_annotation, valtxt_path, savetxt_path, cat_id):
    coco = COCO(annotation_file)
    coco_res = coco.loadRes(res_annotation)

    if not os.path.exists(savetxt_path):
        os.makedirs(savetxt_path)
    else:
        for file in os.listdir(savetxt_path):
            os.remove(os.path.join(savetxt_path, file))

    data = []
    for line in open(valtxt_path, "r"):  # 设置文件对象并读取每一行文件
        data.append(line.strip('\n'))

    image_ids = coco_res.getImgIds()
    for image_id in image_ids:
        image = coco_res.loadImgs(image_id)[0]
        file_name = image['file_name']
        origin_file_name = file_name.split('_')[0] + '_' + file_name.split('_')[1] + '.jpg'
        txt_file_name = file_name.split('.')[0] + ".txt"

        # temp = int(file_name.split('_')[3]) - 640
        temp = int(file_name.split('_')[2]) - 600
        f = open(os.path.join(savetxt_path, txt_file_name), "w")

        annIds = coco_res.getAnnIds(imgIds=image_id, iscrowd=None)
        anns = coco_res.loadAnns(annIds)
        for ann in anns:
            category_id = ann['category_id']
            if category_id == cat_id:
                x, y, w, h = ann['bbox']
                x1 = int(x)
                x2 = int(x + w)
                y1 = int(y)
                y2 = int(y + h)
                color = (0, 0, 255)

                score = ann['score']

                if x2 - x1 >= 5 and y2 - y1 >= 5:
                    f.write(str(x1 + temp) + ',' + str(y1) + ',' + str(x2 + temp) + ',' + str(y2) + ',' + str(
                        round(score, 2)) + '\n')
        f.close()


def hebing_txt(txtPath, saveTxtPath, removeTxtPath, val_txt_path):
    fileroot = os.listdir(saveTxtPath)
    removeList = os.listdir(removeTxtPath)
    for filename in removeList:
        os.remove(os.path.join(removeTxtPath, filename))
    for filename in fileroot:
        os.remove(os.path.join(saveTxtPath, filename))
    data = []
    # for line in open("/mmdetection/data\VOCdevkit\VOC2007\ImageSets\\test.txt", "r"):  # 设置文件对象并读取每一行文件
    for line in open(val_txt_path, "r"):  # 设置文件对象并读取每一行文件
        data.append(line.strip('\n'))
    txtList = os.listdir(txtPath)
    for txtfile in txtList:
        for image in data:
            if image.split('_')[1] == txtfile.split('_')[1]:
                # print(image.split('_')[1])
                fw = open(os.path.join(saveTxtPath, image + '.txt'), 'a')  # w覆盖，a追加
                for line in open(os.path.join(txtPath, txtfile), "r"):  # 设置文件对象并读取每一行文件
                    fw.write(line)
                fw.close()

    fileroot = os.listdir(saveTxtPath)
    for file in fileroot:
        print(file)
        oldname = os.path.join(saveTxtPath, file)
        newname = os.path.join(removeTxtPath, file)
        shutil.copyfile(oldname, newname)  # 将需要的文件从oldname复制到newname
    print("finish")


def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return keep


def plot_bbox(dets, c='k'):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    plt.plot([x1, x2], [y1, y1], c)
    plt.plot([x1, x1], [y1, y2], c)
    plt.plot([x1, x2], [y2, y2], c)
    plt.plot([x2, x2], [y1, y2], c)
    plt.title(" nms")


# plt.figure(1)
# ax1 = plt.subplot(1, 2, 1)
# ax2 = plt.subplot(1, 2, 2)
def nms_box(imagePath, imagesavePath, txtPath, thresh):
    txtList = os.listdir(txtPath)
    for txtfile in tqdm.tqdm(txtList):
        boxes = np.loadtxt(os.path.join(txtPath, txtfile), dtype=np.float32,
                           delimiter=',')
        if boxes.size > 5:
            fw = open(os.path.join(txtPath, txtfile), 'w')
            print(boxes.size)
            # plt.sca(ax1)
            # plot_bbox(boxes, 'k')  # before nms
            print(txtfile)
            keep = py_cpu_nms(boxes, thresh=thresh)
            # print(keep)
            # plt.sca(ax2)
            # plot_bbox(boxes[keep], 'r')  # after nms
            # plt.show()
            img = cv.imread(os.path.join(imagePath, txtfile[:-3] + 'jpg'), 0)
            for label in boxes[keep]:
                fw.write(str(int(label[0])) + ',' + str(int(label[1])) + ',' + str(int(label[2])) + ',' + str(
                    int(label[3])) + ',' + str(round((label[4]), 2)) + '\n')
                Xmin = int(label[0])
                Ymin = int(label[1])
                Xmax = int(label[2])
                Ymax = int(label[3])
                color = (0, 0, 255)
                if Xmax - Xmin >= 5 and Ymax - Ymin >= 5:
                    cv.rectangle(img, (Xmin, Ymin), (Xmax, Ymax), color, 1)
                    font = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(img, str(round((label[4]), 2)), (Xmin, Ymin - 7), font, 0.2, (6, 230, 230),
                               1)
            print(os.path.join(imagesavePath, txtfile[:-3] + 'jpg'))
            cv.imwrite(os.path.join(imagesavePath, txtfile[:-3] + 'jpg'), img)
            fw.close()


def write_huizong(txtPath, saveTxtPath):
    data = []
    txtList = os.listdir(txtPath)
    fw = open(os.path.join(saveTxtPath, 'qikong.txt'), 'w')  # w覆盖，a追加
    for txtfile in txtList:
        for line in open(os.path.join(txtPath, txtfile), 'r'):  # 设置文件对象并读取每一行文件
            line = line.strip('\n')
            fw.write(txtfile[:-4] + ' ' +
                     line.split(',')[4] + ' ' +
                     line.split(',')[0] + ' ' + line.split(',')[1] + ' ' + line.split(',')[2] + ' ' + line.split(',')[
                         3] + '\n')
    fw.close()
    print("finish")


np.seterr(divide='ignore', invalid='ignore')


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):  # voc2007的计算方式和voc2012的计算方式不同，目前一般采用第二种
    """ ap = voc_ap(rec, prec, [use_07_metric])
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


## 程序入口

def voc_eval(detpath,  # 保存检测到的目标框的文件路径，每一类的目标框单独保存在一个文件
             annopath,  # Annotations的路径
             imagesetfile,  # 测试图片名字列表
             classname,  # 类别名称
             cachedir,  # 缓存文件夹
             ovthresh=0.5,  # IoU阈值
             use_07_metric=False):  # mAP计算方法
    """rec, prec, ap = voc_eval(detpath,
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

    # first load gt   获取真实目标框
    # 当程序第一次运行时，会读取Annotations下的xml文件获取每张图片中真实的目标框
    # 然后把获取的结果保存在annotations_cache文件夹中
    # 以后再次运行时直接从缓存文件夹中读取真实目标

    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(annopath.format(imagename))
        if i % 100 == 0:
            print('Reading annotation for {:d}/{:d}'.format(
                i + 1, len(imagenames)))
    # extract gt objects for this class 提取该类的真实目标
    class_recs = {}
    npos = 0  # 保存该类一共有多少真实目标
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]  # 保存名字为imagename的图片中，类别为classname的目标框的信息
        bbox = np.array([x['bbox'] for x in R])  # 目标框的坐标
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)  # 是否是难以识别的目标
        det = [False] * len(R)  # 每一个目标框对应一个det[i]，用来判断该目标框是否已经处理过
        npos = npos + sum(~difficult)  # 计算总的目标个数
        class_recs[imagename] = {'bbox': bbox,  # 把每一张图像中的目标框信息放到class_recs中
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)  # 打开classname类别检测到的目标框文件
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]  # 图像名字
    confidence = np.array([float(x[1]) for x in splitlines])  # 置信度
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])  # 目标框坐标

    # sort by confidence  按照置信度排序
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)  # 统计检测到的目标框个数
    tp = np.zeros(nd)  # 创建tp列表，列表长度为目标框个数
    fp = np.zeros(nd)  # 创建fp列表，列表长度为目标框个数

    for d in range(nd):
        print(nd)
        R = class_recs[image_ids[d]]  # 得到图像名字为image_ids[d]真实的目标框信息
        bb = BB[d, :].astype(float)  # 得到图像名字为image_ids[d]检测的目标框坐标
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)  # 得到图像名字为image_ids[d]真实的目标框坐标

        if BBGT.size > 0:
            # compute overlaps  计算IoU
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)  # 检测到的目标框可能预若干个真实目标框都有交集，选择其中交集最大的
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:  # IoU是否大于阈值
            if not R['difficult'][jmax]:  # 真实目标框是否难以识别
                if not R['det'][jmax]:  # 该真实目标框是否已经统计过
                    tp[d] = 1.  # 将tp对应第d个位置变成1
                    R['det'][jmax] = 1  # 将该真实目标框做标记
                else:
                    fp[d] = 1.  # 否则将fp对应的位置变为1
        else:
            fp[d] = 1.  # 否则将fp对应的位置变为1

    # compute precision recall
    fp = np.cumsum(fp)  # 按列累加，最大值即为tp数量
    tp = np.cumsum(tp)  # 按列累加，最大值即为fp数量
    rec = tp / float(npos)  # 计算recall
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)  # 计算精度
    ap = voc_ap(rec, prec, use_07_metric)  # 计算ap

    return rec, prec, ap