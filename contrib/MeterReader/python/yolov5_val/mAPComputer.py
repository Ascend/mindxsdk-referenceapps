# !/usr/bin/env python
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

import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math
import cv2
import numpy as np

import matplotlib.pyplot as plt

'''
1.map_det.py:得到验证集中数据的预测结果，并且将结果保存成txt，以”图片名.txt“命名，txt中数据格式为：
    <class_id> <x_center> <y_center> <width> <height>
2.yolo2voc.py:将验证集的groundTruth数据和经过模型得到的结果数据的txt文件转换成voc格式
    <class_id> <left> <top> <right> <bottom> <confidence>
3.no_det.py:过滤没有检测到目标的图片文件
4.computer_mAP.py:计算mAp
'''

MINOVERLAP = 0.4

parser = argparse.ArgumentParser()
parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
args = parser.parse_args()

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''

if args.ignore is None:
    args.ignore = []

specificIouFlagged = False
if args.set_class_iou is not None:
    specificIouFlagged = True

os.chdir(os.path.dirname(os.path.abspath(__file__)))

cur_path = os.path.abspath(os.path.dirname(__file__))
GROUND_TRUE_PATH = os.path.join(cur_path, 'det_val_data', 'det_val_voc').replace('\\', '/')
DETECTION_RESULTS_PATH = os.path.join(cur_path, 'det_val_data', 'det_sdk_voc').replace('\\', '/')

IMG_PATH = os.path.join(cur_path, 'det_val_data', 'det_val_img').replace('\\', '/')
TEMP_FILES_PATH = os.path.join(cur_path, 'det_temp_files').replace('\\', '/')
results_files_path = os.path.join(cur_path, 'det_res').replace('\\', '/')

if os.path.exists(IMG_PATH):
    for dirpath, dirnames, files in os.walk(IMG_PATH):
        if not files:
            args.no_animation = True
else:
    args.no_animation = True

showAnimation = False
if not args.no_animation:
    showAnimation = True
else:
    showAnimation = False

drawPlot = False
if not args.no_plot:
    drawPlot = True
else:
    drawPlot = False


def logAveMissRate(precision, false_positives_cumsum, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced falsePositivesPerImage points
            between 10e-2 and 10e0, in log-space.
        output:
                logAveMissRate | log-average miss rate
                missrate | miss rate
                falsePositivesPerImage | false positives per image
        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    if precision.size == 0:
        logAveMissRate = 0
        missRate = 1
        falsePositivesPerImage = 0
        return logAveMissRate, missRate, falsePositivesPerImage

    falsePositivesPerImage = false_positives_cumsum / float(num_images)
    missRate = (1 - precision)

    falsePositivesPerImageTMP = np.insert(falsePositivesPerImage, 0, -1.0)
    missRateTmp = np.insert(missRate, 0, 1.0)

    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        j = np.where(falsePositivesPerImageTMP <= ref_i)[-1][-1]
        ref[i] = missRateTmp[j]

    logAveMissRate = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return logAveMissRate, missRate, falsePositivesPerImage


"""
 throw error and exit
"""


def error(msg):
    print(msg)
    sys.exit(0)


"""
 check if the number is a float between 0.0 and 1.0
"""


def is_float_between_0_and_1(value):
    val = float(value)
    try:        
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""


def vocAp(rec, prec):
    """
    --- Official matlab code VOC2012---
    miss_rateec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(miss_rateec(2:end)~=miss_rateec(1:end-1))+1;
    ap=sum((miss_rateec(i)-miss_rateec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    miss_rateec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(miss_rateec(2:end)~=miss_rateec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(miss_rateec)):
        if miss_rateec[i] != miss_rateec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((miss_rateec(i)-miss_rateec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((miss_rateec[i] - miss_rateec[i - 1]) * mpre[i])
    return ap, miss_rateec, mpre


"""
 Convert the lines of a file to a list
"""


def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


"""
 Draws text in image
"""


def draw_text_in_image(img, text, pos, color, line_width):
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                color,
                lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)


"""
 Plot - adjust axes
"""


def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


"""
 Draw plot using Matplotlib
"""


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                   true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> false_positives: False Positives (object detected but does not match ground-truth)
            - orange -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        false_positives_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            false_positives_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), false_positives_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive',
                 left=false_positives_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            false_positives_val = false_positives_sorted[i]
            tp_val = tp_sorted[i]
            false_positives_str_val = " " + str(false_positives_val)
            tp_str_val = false_positives_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, false_positives_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)  # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15  # in percentage of the figure height
    bottom_margin = 0.05  # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


"""
 Create a ".temp_files/" and "results/" directory
"""

if drawPlot:
    os.makedirs(os.path.join(results_files_path, "AP"))
    os.makedirs(os.path.join(results_files_path, "F1"))
    os.makedirs(os.path.join(results_files_path, "Recall"))
    os.makedirs(os.path.join(results_files_path, "Precision"))
if showAnimation:
    os.makedirs(os.path.join(results_files_path, "images", "detections_one_by_one"))

"""
 ground-truth
     Load each of the ground-truth files into a temporary ".json" file.
     Create a list of all the class names present in the ground-truth (ground_truth_classes).
"""
# get a list with the ground-truth files
ground_truth_files_list = glob.glob(GROUND_TRUE_PATH + '/*.txt')
if len(ground_truth_files_list) == 0:
    error("Error: No ground-truth files found!")
ground_truth_files_list.sort()
# dictionary with counter per class
ground_truth_counter_per_class = {}
counter_images_per_class = {}

for txt_file in ground_truth_files_list:
    # print(txt_file)
    file_id = txt_file.split(".txt", 1)[0]
    file_id = os.path.basename(os.path.normpath(file_id))
    # check if there is a correspondent detection-results file
    temp_path = os.path.join(DETECTION_RESULTS_PATH, (file_id + ".txt"))
    if not os.path.exists(temp_path):
        error_msg = "Error. File not found: {}\n".format(temp_path)
        error_msg += "(You can avoid this error message by running extra/intersect-ground_truth-and-dr.py)"
        error(error_msg)
    lines_list = file_lines_to_list(txt_file)
    # create ground-truth dictionary
    bounding_boxes = []
    is_difficult = False
    already_seen_classes = []
    for line in lines_list:
        try:
            if "difficult" in line:
                class_name, left, top, right, bottom, _difficult = line.split()
                is_difficult = True
            else:
                class_name, left, top, right, bottom = line.split()

        except:
            if "difficult" in line:
                line_split = line.split()
                _difficult = line_split[-1]
                bottom = line_split[-2]
                right = line_split[-3]
                top = line_split[-4]
                left = line_split[-5]
                class_name = ""
                for name in line_split[:-5]:
                    class_name += name + " "
                class_name = class_name[:-1]
                is_difficult = True
            else:
                line_split = line.split()
                bottom = line_split[-1]
                right = line_split[-2]
                top = line_split[-3]
                left = line_split[-4]
                class_name = ""
                for name in line_split[:-4]:
                    class_name += name + " "
                class_name = class_name[:-1]
        if class_name in args.ignore:
            continue
        bbox = left + " " + top + " " + right + " " + bottom
        if is_difficult:
            bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
            is_difficult = False
        else:
            bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
            if class_name in ground_truth_counter_per_class:
                ground_truth_counter_per_class[class_name] += 1
            else:
                ground_truth_counter_per_class[class_name] = 1

            if class_name not in already_seen_classes:
                if class_name in counter_images_per_class:
                    counter_images_per_class[class_name] += 1
                else:
                    counter_images_per_class[class_name] = 1
                already_seen_classes.append(class_name)

    with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
        json.dump(bounding_boxes, outfile)

ground_truth_classes = list(ground_truth_counter_per_class.keys())
ground_truth_classes = sorted(ground_truth_classes)
n_classes = len(ground_truth_classes)

"""
 Check format of the flag --set-class-iou (if used)
    e.g. check if class exists
"""
if specificIouFlagged:
    n_args = len(args.set_class_iou)
    error_msg = \
        '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
    if n_args % 2 != 0:
        error('Error, missing arguments. Flag usage:' + error_msg)
    # [class_1] [IoU_1] [class_2] [IoU_2]
    # specific_iou_classes = ['class_1', 'class_2']
    specific_iou_classes = args.set_class_iou[::2]  # even
    # iou_list = ['IoU_1', 'IoU_2']
    iou_list = args.set_class_iou[1::2]  # odd
    if len(specific_iou_classes) != len(iou_list):
        error('Error, missing arguments. Flag usage:' + error_msg)
    for tmp_class in specific_iou_classes:
        if tmp_class not in ground_truth_classes:
            error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
    for num in iou_list:
        if not is_float_between_0_and_1(num):
            error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

"""
 detection-results
     Load each of the detection-results files into a temporary ".json" file.
"""
dr_files_list = glob.glob(DETECTION_RESULTS_PATH + '/*.txt')
dr_files_list.sort()

for class_index, class_name in enumerate(ground_truth_classes):
    bounding_boxes = []
    for txt_file in dr_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        temp_path = os.path.join(GROUND_TRUE_PATH, (file_id + ".txt"))
        if class_index == 0:
            if not os.path.exists(temp_path):
                error_msg = "Error. File not found: {}\n".format(temp_path)
                error_msg += "(You can avoid this error message by running extra/intersect-ground_truth-and-dr.py)"
                error(error_msg)
        lines = file_lines_to_list(txt_file)
        for line in lines:
            try:
                tmp_class_name, confidence, left, top, right, bottom = line.split()
            except:
                line_split = line.split()
                bottom = line_split[-1]
                right = line_split[-2]
                top = line_split[-3]
                left = line_split[-4]
                confidence = line_split[-5]
                tmp_class_name = ""
                for name in line_split[:-5]:
                    tmp_class_name += name + " "
                tmp_class_name = tmp_class_name[:-1]

            if tmp_class_name == class_name:
                bbox = left + " " + top + " " + right + " " + bottom
                bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})

    bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
    with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
        json.dump(bounding_boxes, outfile)

"""
 Calculate the AP for each class
"""
sum_AP = 0.0
apDictionary = {}
logAveMissRateDictionary = {}
with open(results_files_path + "/results.txt", 'w') as results_file:
    results_file.write("# AP and precision/recall per class\n")
    count_true_positives = {}

    for class_index, class_name in enumerate(ground_truth_classes):
        count_true_positives[class_name] = 0
        """
         Load detection-results of that class
        """
        dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
        dr_data = json.load(open(dr_file))
        """
         Assign detection-results to ground-truth objects
        """
        nd = len(dr_data)
        tp = [0] * nd
        false_positives = [0] * nd
        score = [0] * nd
        score05_idx = 0
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            score[idx] = float(detection["confidence"])
            if score[idx] > 0.5:
                score05_idx = idx

            if showAnimation:
                ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
                if len(ground_truth_img) == 0:
                    error("Error. Image not found with id: " + file_id)
                elif len(ground_truth_img) > 1:
                    error("Error. Multiple image with id: " + file_id)
                else:
                    img = cv2.imiss_rateead(IMG_PATH + "/" + ground_truth_img[0])
                    img_cumulative_path = results_files_path + "/images/" + ground_truth_img[0]
                    if os.path.isfile(img_cumulative_path):
                        img_cumulative = cv2.imiss_rateead(img_cumulative_path)
                    else:
                        img_cumulative = img.copy()
                    bottom_border = 60
                    BLACK = [0, 0, 0]
                    img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)

            ground_truth_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(ground_truth_file))
            ovmax = -1
            ground_truth_match = -1
            bb = [float(x) for x in detection["bbox"].split()]
            for obj in ground_truth_data:
                if obj["class_name"] == class_name:
                    bbground_truth = [float(x) for x in obj["bbox"].split()]
                    bi = [max(bb[0], bbground_truth[0]), max(bb[1], bbground_truth[1]), min(bb[2], bbground_truth[2]),
                          min(bb[3], bbground_truth[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbground_truth[2] - bbground_truth[0]
                                                                          + 1) * (
                                         bbground_truth[3] - bbground_truth[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            ground_truth_match = obj

            if showAnimation:
                status = "NO MATCH FOUND!"
            min_overlap = MINOVERLAP
            if specificIouFlagged:
                if class_name in specific_iou_classes:
                    index = specific_iou_classes.index(class_name)
                    min_overlap = float(iou_list[index])
            if ovmax >= min_overlap:
                if "difficult" not in ground_truth_match:
                    if not bool(ground_truth_match["used"]):
                        tp[idx] = 1
                        ground_truth_match["used"] = True
                        count_true_positives[class_name] += 1
                        with open(ground_truth_file, 'w') as f:
                            f.write(json.dumps(ground_truth_data))
                        if showAnimation:
                            status = "MATCH!"
                    else:
                        false_positives[idx] = 1
                        if showAnimation:
                            status = "REPEATED MATCH!"
            else:
                false_positives[idx] = 1
                if ovmax > 0:
                    status = "INSUFFICIENT OVERLAP"

            """
             Draw image to show animation
            """
            if showAnimation:
                height, widht = img.shape[:2]
                # colors (OpenCV works with BGR)
                white = (255, 255, 255)
                light_blue = (255, 200, 100)
                green = (0, 255, 0)
                light_red = (30, 30, 255)
                # 1st line
                margin = 10
                v_pos = int(height - margin - (bottom_border / 2.0))
                text = "Image: " + ground_truth_img[0] + " "
                img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
                img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
                if ovmax != -1:
                    color = light_red
                    if status == "INSUFFICIENT OVERLAP":
                        text = "IoU: {0:.2f}% ".format(ovmax * 100) + "< {0:.2f}% ".format(min_overlap * 100)
                    else:
                        text = "IoU: {0:.2f}% ".format(ovmax * 100) + ">= {0:.2f}% ".format(min_overlap * 100)
                        color = green
                    img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                # 2nd line
                v_pos += int(bottom_border / 2.0)
                rank_pos = str(idx + 1)  # rank position (idx starts at 0)
                text = "Detection #rank: " + rank_pos + " confidence: {0:.2f}% ".format(
                    float(detection["confidence"]) * 100)
                img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                color = light_red
                if status == "MATCH!":
                    color = green
                text = "Result: " + status + " "
                img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

                font = cv2.FONT_HERSHEY_SIMPLEX
                if ovmax > 0:  # if there is intersections between the bounding-boxes
                    bbground_truth = [int(round(float(x))) for x in ground_truth_match["bbox"].split()]
                    cv2.rectangle(img, (bbground_truth[0], bbground_truth[1]), (bbground_truth[2], bbground_truth[3]),
                                  light_blue, 2)
                    cv2.rectangle(img_cumulative, (bbground_truth[0], bbground_truth[1]),
                                  (bbground_truth[2], bbground_truth[3]), light_blue, 2)
                    cv2.putText(img_cumulative, class_name, (bbground_truth[0], bbground_truth[1] - 5), font, 0.6,
                                light_blue, 1,
                                cv2.LINE_AA)
                bb = [int(i) for i in bb]
                cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                cv2.rectangle(img_cumulative, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                cv2.putText(img_cumulative, class_name, (bb[0], bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
                # show image
                cv2.imshow("Animation", img)
                cv2.waitKey(20)  # show for 20 ms
                # save image to results
                output_img_path = results_files_path + "/images/detections_one_by_one/" + class_name + "_detection" + str(
                    idx) + ".jpg"
                cv2.imwrite(output_img_path, img)
                # save the image with all the objects drawn to it
                cv2.imwrite(img_cumulative_path, img_cumulative)

        cumsum = 0
        for idx, val in enumerate(false_positives):
            false_positives[idx] += cumsum
            cumsum += val

        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val

        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / np.maximum(ground_truth_counter_per_class[class_name], 1)

        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / np.maximum((false_positives[idx] + tp[idx]), 1)

        ap, miss_rateec, mprec = vocAp(rec[:], prec[:])
        F1 = np.array(rec) * np.array(prec) * 2 / np.where((np.array(prec) + np.array(rec)) == 0, 1,
                                                           (np.array(prec) + np.array(rec)))

        sum_AP += ap
        text = "{0:.2f}%".format(ap * 100) + " = " + class_name + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)

        if len(prec) > 0:
            F1_text = "{0:.2f}".format(F1[score05_idx]) + " = " + class_name + " F1 "
            Recall_text = "{0:.2f}%".format(rec[score05_idx] * 100) + " = " + class_name + " Recall "
            Precision_text = "{0:.2f}%".format(prec[score05_idx] * 100) + " = " + class_name + " Precision "
        else:
            F1_text = "0.00" + " = " + class_name + " F1 "
            Recall_text = "0.00%" + " = " + class_name + " Recall "
            Precision_text = "0.00%" + " = " + class_name + " Precision "

        rounded_prec = ['%.2f' % elem for elem in prec]
        rounded_rec = ['%.2f' % elem for elem in rec]
        results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
        if not args.quiet:
            if len(prec) > 0:
                print(text + "\t||\tscore_threhold=0.5 : " + "F1=" + "{0:.2f}".format(F1[score05_idx]) \
                      + " ; Recall=" + "{0:.2f}%".format(rec[score05_idx] * 100) + " ; Precision=" + "{0:.2f}%".format(
                    prec[score05_idx] * 100))
            else:
                print(text + "\t||\tscore_threhold=0.5 : F1=0.00% ; Recall=0.00% ; Precision=0.00%")
        apDictionary[class_name] = ap

        n_images = counter_images_per_class[class_name]
        logAveMissRate, missRate, falsePositivesPerImage = logAveMissRate(np.array(rec),
                                                                                    np.array(false_positives), n_images)
        logAveMissRateDictionary[class_name] = logAveMissRate

        """
         Draw plot
        """
        if drawPlot:
            plt.plot(rec, prec, '-o')
            area_under_curve_x = miss_rateec[:-1] + [miss_rateec[-2]] + [miss_rateec[-1]]
            area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
            plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')

            fig = plt.gcf()
            fig.canvas.set_window_title('AP ' + class_name)

            plt.title('class: ' + text)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            axes = plt.gca()
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.05])
            fig.savefig(results_files_path + "/AP/" + class_name + ".png")
            plt.cla()

            plt.plot(score, F1, "-", color='orangered')
            plt.title('class: ' + F1_text + "\nscore_threhold=0.5")
            plt.xlabel('Score_Threhold')
            plt.ylabel('F1')
            axes = plt.gca()
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.05])
            fig.savefig(results_files_path + "/F1/" + class_name + ".png")
            plt.cla()

            plt.plot(score, rec, "-H", color='gold')
            plt.title('class: ' + Recall_text + "\nscore_threhold=0.5")
            plt.xlabel('Score_Threhold')
            plt.ylabel('Recall')
            axes = plt.gca()
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.05])
            fig.savefig(results_files_path + "/Recall/" + class_name + ".png")
            plt.cla()

            plt.plot(score, prec, "-s", color='palevioletred')
            plt.title('class: ' + Precision_text + "\nscore_threhold=0.5")
            plt.xlabel('Score_Threhold')
            plt.ylabel('Precision')
            axes = plt.gca()
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.05])
            fig.savefig(results_files_path + "/Precision/" + class_name + ".png")
            plt.cla()

    if showAnimation:
        cv2.destroyAllWindows()

    results_file.write("\n# mAP of all classes\n")
    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP * 100)
    results_file.write(text + "\n")
    print(text)

# remove the temp_files directory
shutil.rmtree(TEMP_FILES_PATH)

"""
 Count total of detection-results
"""
# iterate through all the files
det_counter_per_class = {}
for txt_file in dr_files_list:
    # get lines to list
    lines_list = file_lines_to_list(txt_file)
    for line in lines_list:
        class_name = line.split()[0]
        # check if class is in the ignore list, if yes skip
        if class_name in args.ignore:
            continue
        # count that object
        if class_name in det_counter_per_class:
            det_counter_per_class[class_name] += 1
        else:
            # if class didn't exist yet
            det_counter_per_class[class_name] = 1
# print(det_counter_per_class)
dr_classes = list(det_counter_per_class.keys())

"""
 Plot the total number of occurences of each class in the ground-truth
"""
if drawPlot:
    window_title = "ground-truth-info"
    plot_title = "ground-truth\n"
    plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
    x_label = "Number of objects per class"
    output_path = results_files_path + "/ground-truth-info.png"
    to_show = False
    plot_color = 'forestgreen'
    draw_plot_func(
        ground_truth_counter_per_class,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        '',
    )

"""
 Write number of ground-truth objects per class to results.txt
"""
with open(results_files_path + "/results.txt", 'a') as results_file:
    results_file.write("\n# Number of ground-truth objects per class\n")
    for class_name in sorted(ground_truth_counter_per_class):
        results_file.write(class_name + ": " + str(ground_truth_counter_per_class[class_name]) + "\n")

"""
 Finish counting true positives
"""
for class_name in dr_classes:
    # if class exists in detection-result but not in ground-truth then there are no true positives in that class
    if class_name not in ground_truth_classes:
        count_true_positives[class_name] = 0
# print(count_true_positives)

"""
 Plot the total number of occurences of each class in the "detection-results" folder
"""
if drawPlot:
    window_title = "detection-results-info"
    # Plot title
    plot_title = "detection-results\n"
    plot_title += "(" + str(len(dr_files_list)) + " files and "
    count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
    plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
    # end Plot title
    x_label = "Number of objects per class"
    output_path = results_files_path + "/detection-results-info.png"
    to_show = False
    plot_color = 'forestgreen'
    true_p_bar = count_true_positives
    draw_plot_func(
        det_counter_per_class,
        len(det_counter_per_class),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        true_p_bar
    )

"""
 Write number of detected objects per class to results.txt
"""
with open(results_files_path + "/results.txt", 'a') as results_file:
    results_file.write("\n# Number of detected objects per class\n")
    for class_name in sorted(dr_classes):
        n_det = det_counter_per_class[class_name]
        text = class_name + ": " + str(n_det)
        text += " (tp:" + str(count_true_positives[class_name]) + ""
        text += ", false_positives:" + str(n_det - count_true_positives[class_name]) + ")\n"
        results_file.write(text)

"""
 Draw log-average miss rate plot (Show logAveMissRate of all classes in decreasing order)
"""
if drawPlot:
    window_title = "logAveMissRate"
    plot_title = "log-average miss rate"
    x_label = "log-average miss rate"
    output_path = results_files_path + "/log_ave_miss_rate.png"
    to_show = False
    plot_color = 'royalblue'
    draw_plot_func(
        logAveMissRateDictionary,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
    )

"""
 Draw mAP plot (Show AP's of all classes in decreasing order)
"""
if drawPlot:
    window_title = "mAP"
    plot_title = "mAP = {0:.2f}%".format(mAP * 100)
    x_label = "Average Precision"
    output_path = results_files_path + "/mAP.png"
    to_show = True
    plot_color = 'royalblue'
    draw_plot_func(
        apDictionary,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
    )
