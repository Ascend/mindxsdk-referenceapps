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
import argparse

parser = argparse.ArgumentParser(description="FasterRcnn evaluation")
parser.add_argument("--ann_file", type=str,
                    default="../data/eval/cocodataset/annotations/instances_val2017.json",
                    help="Ann file, default is val.json.")
parser.add_argument("--result_json_file", type=str,
                    default="./results.pkl.bbox.json",
                    required=False, help="results.pkl.bbox.json file path.")
parser.add_argument("--voc_dir", type=str, default="../data/eval/VOCdevkit/",
                    help="VOCdevkit file path.")
parser.add_argument("--cat_id", type=int, default=1, help="Category id, default is 1.")
parser.add_argument("--object_name", type=str, default="qikong", help="the object to eval")
args_opt = parser.parse_args()


def VOC_eval(ann_file, result_json_file, voc_dir, cat_id, object_name):
    TXT_SAVE_PATH = os.path.join(voc_dir, "VOC2017/image_txt")
    VAL_TXT_PATH = os.path.join(voc_dir, "VOC2017/ImageSets/Main/val.txt")
    coco_to_txt(ann_file, result_json_file, VAL_TXT_PATH, TXT_SAVE_PATH, cat_id=cat_id)

    TXT_PATH = TXT_SAVE_PATH
    ALL_TXT_PATH = os.path.join(voc_dir, "VOC2017/image_huizong_txt")
    NMS_TXT_PATH = os.path.join(voc_dir, "VOC2017/image_huizong_txt_nms")
    CUT_PATH = VAL_TXT_PATH
    hebing_txt(TXT_PATH, ALL_TXT_PATH, NMS_TXT_PATH, CUT_PATH)

    CUT_PATH = os.path.join(voc_dir, "VOC2017/JPEGImages")
    imagesavePath = os.path.join(voc_dir, "VOC2017/images1")
    TXT_PATH = NMS_TXT_PATH
    nms_box(CUT_PATH, imagesavePath, TXT_PATH, thresh=0.1)

    TXT_PATH = NMS_TXT_PATH
    ALL_TXT_PATH = os.path.join(voc_dir, "VOC2017/obj_txt_huizong")
    write_huizong(TXT_PATH, ALL_TXT_PATH)

    aps = []
    recs = []
    precs = []
    ANNO_PATH = voc_dir + "/VOC2017/Annotations/" + '{:s}.xml'
    imagesetfile = os.path.join(voc_dir, "VOC2017/ImageSets/Main/val.txt")
    cachedir = os.path.join(voc_dir, "VOC2017/demo")
    filename = os.path.join(voc_dir, "VOC2017/obj_txt_huizong/qikong.txt")

    rec, prec, ap = voc_eval(
        filename, ANNO_PATH, imagesetfile, object_name, cachedir, ovthresh=0.5,
        use_07_metric=False)

    aps += [ap]

    print('AP for {} = {:.4f}'.format(object_name, ap))
    print('recall for {} = {:.4f}'.format(object_name, rec[-1]))
    print('precision for {} = {:.4f}'.format(object_name, prec[-1]))
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
    for line in open(valtxt_path, "r"):
        data.append(line.strip('\n'))

    image_ids = coco_res.getImgIds()
    for image_id in image_ids:
        image = coco_res.loadImgs(image_id)[0]
        file_name = image['file_name']
        origin_file_name = file_name.split('_')[0] + '_' + file_name.split('_')[1] + '.jpg'
        txt_file_name = file_name.split('.')[0] + ".txt"

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
    for line in open(val_txt_path, "r"):
        data.append(line.strip('\n'))
    txtList = os.listdir(txtPath)
    for txtfile in txtList:
        for image in data:
            if image.split('_')[1] == txtfile.split('_')[1]:
                fw = open(os.path.join(saveTxtPath, image + '.txt'), 'a')
                for line in open(os.path.join(txtPath, txtfile), "r"):
                    fw.write(line)
                fw.close()

    fileroot = os.listdir(saveTxtPath)
    for file in fileroot:
        print(file)
        oldname = os.path.join(saveTxtPath, file)
        newname = os.path.join(removeTxtPath, file)
        shutil.copyfile(oldname, newname)
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


def nms_box(imagePath, imagesavePath, txtPath, thresh):
    txtList = os.listdir(txtPath)
    for txtfile in tqdm.tqdm(txtList):
        boxes = np.loadtxt(os.path.join(txtPath, txtfile), dtype=np.float32,
                           delimiter=',')
        if boxes.size > 5:
            fw = open(os.path.join(txtPath, txtfile), 'w')
            print(boxes.size)
            print(txtfile)
            keep = py_cpu_nms(boxes, thresh=thresh)
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
    fw = open(os.path.join(saveTxtPath, 'qikong.txt'), 'w')
    for txtfile in txtList:
        for line in open(os.path.join(txtPath, txtfile), 'r'):
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


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
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
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        print(nd)
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
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
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

            # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


if __name__ == '__main__':
    VOC_eval(args_opt.ann_file, args_opt.result_json_file, args_opt.voc_dir, args_opt.cat_id, args_opt.object_name)
