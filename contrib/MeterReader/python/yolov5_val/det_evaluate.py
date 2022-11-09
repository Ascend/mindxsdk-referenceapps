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
import stat
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
4.det_evaluate.py:计算mAp
'''

MINOVERLAP = 0.4
MODES = stat.S_IWUSR | stat.S_IRUSR

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

specificFlagged = False
if args.set_class_iou is not None:
    specificFlagged = True

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

animation = False
if not args.no_animation:
    animation = True
else:
    animation = False

plot = False
if not args.no_plot:
    plot = True
else:
    plot = False


def LogAveMissRate(precision, false_positives_cumsum, num_images):
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
        log_average_miss_rate = 0
        miss_rate = 1
        false_positives_image = 0
        return log_average_miss_rate, miss_rate, false_positives_image

    false_positives_image = false_positives_cumsum / float(num_images)
    miss_rate = (1 - precision)

    false_positives_image_temp = np.insert(false_positives_image, 0, -1.0)
    missRateTmp = np.insert(miss_rate, 0, 1.0)

    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        j = np.where(false_positives_image_temp <= ref_i)[-1][-1]
        ref[i] = missRateTmp[j]

    log_average_miss_rate = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return log_average_miss_rate, miss_rate, false_positives_image


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
    value_temp = float(value)
    try:        
        if value_temp > 0.0 and value_temp < 1.0:
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


def voc_ap(rec_data, prec_data):
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
    rec_data.insert(0, 0.0)  # insert 0.0 at begining of list
    rec_data.append(1.0)  # insert 1.0 at end of list
    miss_rateec_data = rec_data[:]
    prec_data.insert(0, 0.0)  # insert 0.0 at begining of list
    prec_data.append(0.0)  # insert 0.0 at end of list
    mpre = prec_data[:]
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
        matlab: i=find(miss_rateec_data(2:end)~=miss_rateec_data(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(miss_rateec_data)):
        if miss_rateec_data[i] != miss_rateec_data[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((miss_rateec(i)-miss_rateec(i-1)).*mpre(i));
    """
    ap_data = 0.0
    for i in i_list:
        ap_data += ((miss_rateec_data[i] - miss_rateec_data[i - 1]) * mpre[i])
    return ap_data, miss_rateec_data, mpre


"""
 Convert the lines of a file to a list
"""


def file_lines_to_list(file_path):
    # open txt file lines to a list
    with os.fdopen(os.open(file_path, os.O_RDONLY, MODES), 'r') as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


"""
 Draws text in image
"""


def draw_text_in_image(draw_img, draw_text, pos, draw_color, draw_line_width):
    draw_font = cv2.FONT_HERSHEY_PLAIN
    draw_font_scale = 1
    line_type = 1
    bottom_left_corner_of_text = pos
    cv2.putText(draw_img, draw_text,
                bottom_left_corner_of_text,
                draw_font,
                draw_font_scale,
                draw_color,
                line_type)
    text_width, _ = cv2.getTextSize(draw_text, draw_font, draw_font_scale, line_type)[0]
    return draw_img, (draw_line_width + text_width)


"""
 Plot - adjust axes
"""


def adjust_axes(r, t, adjust_fig, adjust_axes):
    # get text width for re-scaling
    adjust_bb = t.get_window_extent(renderer=r)
    text_width_inches = adjust_bb.width / adjust_fig.dpi
    # get axis width in inches
    current_fig_width = adjust_fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = adjust_axes.get_xlim()
    adjust_axes.set_xlim([x_lim[0], x_lim[1] * propotion])


"""
 Draw plot using Matplotlib
"""


def draw_plot_func(dictionary, draw_plot_n_classes, draw_plot_window_title, draw_plot_plot_title, draw_plot_x_label, draw_plot_output_path, to_show, draw_plot_plot_color,
                   draw_plot_true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if draw_plot_true_p_bar != "":
        """
         Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> false_positives: False Positives (object detected but does not match ground-truth)
            - orange -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        false_positives_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            false_positives_sorted.append(dictionary[key] - draw_plot_true_p_bar[key])
            tp_sorted.append(draw_plot_true_p_bar[key])
        plt.barh(range(draw_plot_n_classes), false_positives_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(draw_plot_n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive',
                 left=false_positives_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        draw_plot_fig = plt.gcf()  # gcf - get current figure
        draw_plot_axes = plt.gca()
        r = draw_plot_fig.canvas.get_renderer()
        for i, sorted_val in enumerate(sorted_values):
            false_positives_val = false_positives_sorted[i]
            tp_val = tp_sorted[i]
            false_positives_str_val = " " + str(false_positives_val)
            tp_str_val = false_positives_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(sorted_val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(sorted_val, i, false_positives_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, draw_plot_fig, draw_plot_axes)
    else:
        plt.barh(range(draw_plot_n_classes), sorted_values, color=draw_plot_plot_color)
        """
         Write number on side of bar
        """
        draw_plot_fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = draw_plot_fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)  # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=draw_plot_plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, draw_plot_fig, draw_plot_axes)
    # set window title
    draw_plot_fig.canvas.set_window_title(draw_plot_window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(draw_plot_n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = draw_plot_fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = draw_plot_fig.dpi
    height_pt = draw_plot_n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15  # in percentage of the figure height
    bottom_margin = 0.05  # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        draw_plot_fig.set_figheight(figure_height)

    # set plot title
    plt.title(draw_plot_plot_title, fontsize=14)
    plt.xlabel(draw_plot_x_label, fontsize='large')
    # adjust size of window
    draw_plot_fig.tight_layout()
    # save the plot
    draw_plot_fig.savefig(draw_plot_output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


"""
 Create a ".temp_files/" and "results/" directory
"""

if plot:
    os.makedirs(os.path.join(results_files_path, "AP"))
    os.makedirs(os.path.join(results_files_path, "F1"))
    os.makedirs(os.path.join(results_files_path, "Recall"))
    os.makedirs(os.path.join(results_files_path, "Precision"))
if animation:
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
    file_id = txt_file.split(".txt", 1)[0]
    file_id = os.path.basename(os.path.normpath(file_id))
    # check if there is a correspondent detection-results file
    temp_path = os.path.join(DETECTION_RESULTS_PATH, (file_id + ".txt"))
    if not os.path.exists(temp_path):
        error_text = "Error. File not found: {}\n".format(temp_path)
        error_text += "(You can avoid this error message by running extra/intersect-ground_truth-and-dr.py)"
        error(error_text)
    lines_list = file_lines_to_list(txt_file)
    # create ground-truth dictionary
    bounding_boxes = []
    difficult_flag = False
    already_seen_classes = []
    for line in lines_list:
        try:
            if "difficult" in line:
                className, left, top, right, bottom, _difficult = line.split()
                difficult_flag = True
            else:
                className, left, top, right, bottom = line.split()

        except:
            if "difficult" in line:
                line_split = line.split()
                _difficult = line_split[-1]
                bottom = line_split[-2]
                right = line_split[-3]
                top = line_split[-4]
                left = line_split[-5]
                className = ""
                for name in line_split[:-5]:
                    className += name + " "
                className = className[:-1]
                difficult_flag = True
            else:
                line_split = line.split()
                bottom = line_split[-1]
                right = line_split[-2]
                top = line_split[-3]
                left = line_split[-4]
                className = ""
                for name in line_split[:-4]:
                    className += name + " "
                className = className[:-1]
        if className in args.ignore:
            continue
        bbox = left + " " + top + " " + right + " " + bottom
        if difficult_flag:
            bounding_boxes.append({"class_name": className, "bbox": bbox, "used": False, "difficult": True})
            difficult_flag = False
        else:
            bounding_boxes.append({"class_name": className, "bbox": bbox, "used": False})
            if className in ground_truth_counter_per_class:
                ground_truth_counter_per_class[className] += 1
            else:
                ground_truth_counter_per_class[className] = 1

            if className not in already_seen_classes:
                if className in counter_images_per_class:
                    counter_images_per_class[className] += 1
                else:
                    counter_images_per_class[className] = 1
                already_seen_classes.append(className)

    with os.fdopen(os.open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", os.O_WRONLY | os.O_CREAT, MODES), 'w') as f:
        json.dump(bounding_boxes, f)


ground_truth_classes = list(ground_truth_counter_per_class.keys())
ground_truth_classes = sorted(ground_truth_classes)
n_classes = len(ground_truth_classes)

"""
 Check format of the flag --set-class-iou (if used)
    e.g. check if class exists
"""
if specificFlagged:
    n_args = len(args.set_class_iou)
    error_text = \
        '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
    if n_args % 2 != 0:
        error('Error, missing arguments. Flag usage:' + error_text)

    specific_iou_classes = args.set_class_iou[::2]  # even

    iou_list = args.set_class_iou[1::2]  # odd
    if len(specific_iou_classes) != len(iou_list):
        error('Error, missing arguments. Flag usage:' + error_text)
    for tmp_class in specific_iou_classes:
        if tmp_class not in ground_truth_classes:
            error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_text)
    for num in iou_list:
        if not is_float_between_0_and_1(num):
            error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_text)

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
                error_text = "Error. File not found: {}\n".format(temp_path)
                error_text += "(You can avoid this error message by running extra/intersect-ground_truth-and-dr.py)"
                error(error_text)
        lines = file_lines_to_list(txt_file)
        for line in lines:
            try:
                tmpeClassName, confidence, left, top, right, bottom = line.split()
            except:
                line_split = line.split()
                bottom = line_split[-1]
                right = line_split[-2]
                top = line_split[-3]
                left = line_split[-4]
                confidence = line_split[-5]
                tmpeClassName = ""
                for name in line_split[:-5]:
                    tmpeClassName += name + " "
                tmpeClassName = tmpeClassName[:-1]

            if tmpeClassName == class_name:
                bbox = left + " " + top + " " + right + " " + bottom
                bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})

    bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)

    with os.fdopen(os.open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", os.O_WRONLY | os.O_CREAT, MODES), 'w') as f:
        json.dump(bounding_boxes, f)

"""
 Calculate the AP for each class
"""
sumAp = 0.0
apDictionary = {}
logAveMissRateDictionary = {}
with os.fdopen(os.open(results_files_path + "/results.txt", OSFLAGS, MODES), 'w') as results_file:
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
        scoreIndex = 0
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            score[idx] = float(detection["confidence"])
            if score[idx] > 0.5:
                scoreIndex = idx

            if animation:
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
                    bottomBorder = 60
                    BLACK = [0, 0, 0]
                    img = cv2.copyMakeBorder(img, 0, bottomBorder, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)

            ground_truth_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(ground_truth_file))
            _max = -1
            groundTruthMatch = -1
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
                        if ov > _max:
                            _max = ov
                            groundTruthMatch = obj

            if animation:
                statusFlag = "NO MATCH FOUND!"
            minOverlap = MINOVERLAP
            if specificFlagged:
                if class_name in specific_iou_classes:
                    index = specific_iou_classes.index(class_name)
                    minOverlap = float(iou_list[index])
            if _max >= minOverlap:
                if "difficult" not in groundTruthMatch:
                    if not bool(groundTruthMatch["used"]):
                        tp[idx] = 1
                        groundTruthMatch["used"] = True
                        count_true_positives[class_name] += 1
                        with os.fdopen(os.open(ground_truth_file, os.O_WRONLY, MODES), 'w') as f:
                            f.write(json.dumps(ground_truth_data))
                        if animation:
                            statusFlag = "MATCH!"
                    else:
                        false_positives[idx] = 1
                        if animation:
                            statusFlag = "REPEATED MATCH!"
            else:
                false_positives[idx] = 1
                if _max > 0:
                    statusFlag = "INSUFFICIENT OVERLAP"

            """
             Draw image to show animation
            """
            if animation:
                height, widht = img.shape[:2]
                # colors (OpenCV works with BGR)
                white = (255, 255, 255)
                light_blue = (255, 200, 100)
                green = (0, 255, 0)
                light_red = (30, 30, 255)
                # 1st line
                draw_margin = 10
                v_pos = int(height - draw_margin - (bottomBorder / 2.0))
                image_text = "Image: " + ground_truth_img[0] + " "
                img, line_width = draw_text_in_image(img, text, (draw_margin, v_pos), white, 0)
                image_text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
                img, line_width = draw_text_in_image(img, text, (draw_margin + line_width, v_pos), light_blue, line_width)
                if _max != -1:
                    color = light_red
                    if statusFlag == "INSUFFICIENT OVERLAP":
                        image_text = "IoU: {0:.2f}% ".format(_max * 100) + "< {0:.2f}% ".format(minOverlap * 100)
                    else:
                        image_text = "IoU: {0:.2f}% ".format(_max * 100) + ">= {0:.2f}% ".format(minOverlap * 100)
                        color = green
                    img, _ = draw_text_in_image(img, image_text, (draw_margin + line_width, v_pos), color, line_width)
                # 2nd line
                v_pos += int(bottomBorder / 2.0)
                rank_position = str(idx + 1)  # rank position (idx starts at 0)
                image_text = "Detection #rank: " + rank_position + " confidence: {0:.2f}% ".format(
                    float(detection["confidence"]) * 100)
                img, line_width = draw_text_in_image(img, image_text, (draw_margin, v_pos), white, 0)
                color = light_red
                if statusFlag == "MATCH!":
                    color = green
                image_text = "Result: " + statusFlag + " "
                img, line_width = draw_text_in_image(img, image_text, (draw_margin + line_width, v_pos), color, line_width)

                font = cv2.FONT_HERSHEY_SIMPLEX
                if _max > 0:  # if there is intersections between the bounding-boxes
                    bbground_truth = [int(round(float(x))) for x in groundTruthMatch["bbox"].split()]
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
                output_img_path = results_files_path + "/images/detections_one_by_one/" 
                output_img_path = output_img_path + class_name + "_detection" + str(idx) + ".jpg"
                cv2.imwrite(output_img_path, img)
                # save the image with all the objects drawn to it
                cv2.imwrite(img_cumulative_path, img_cumulative)

        cumcumulative_sum = 0
        for idx, val in enumerate(false_positives):
            false_positives[idx] += cumcumulative_sum
            cumcumulative_sum += val

        cumcumulative_sum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumcumulative_sum
            cumcumulative_sum += val

        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / np.maximum(ground_truth_counter_per_class.get(class_name), 1)

        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / np.maximum((false_positives[idx] + tp[idx]), 1)

        ap, miss_rateec, mprec = voc_ap(rec[:], prec[:])
        F1 = np.array(rec) * np.array(prec) * 2 / np.where((np.array(prec) + np.array(rec)) == 0, 1,
                                                           (np.array(prec) + np.array(rec)))

        sumAp += ap
        image_text = "{0:.2f}%".format(ap * 100) + " = " + class_name + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)

        if len(prec) > 0:
            F1_text = "{0:.2f}".format(F1[scoreIndex]) + " = " + class_name + " F1 "
            Recall_text = "{0:.2f}%".format(rec[scoreIndex] * 100) + " = " + class_name + " Recall "
            Precision_text = "{0:.2f}%".format(prec[scoreIndex] * 100) + " = " + class_name + " Precision "
        else:
            F1_text = "0.00" + " = " + class_name + " F1 "
            Recall_text = "0.00%" + " = " + class_name + " Recall "
            Precision_text = "0.00%" + " = " + class_name + " Precision "

        rounded_prec = ['%.2f' % elem for elem in prec]
        rounded_rec = ['%.2f' % elem for elem in rec]
        results_file.write(image_text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
        if not args.quiet:
            if len(prec) > 0:
                print(image_text + "\t||\tscore_threhold=0.5 : " + "F1=" + "{0:.2f}".format(F1[scoreIndex]) \
                      + " ; Recall=" + "{0:.2f}%".format(rec[scoreIndex] * 100) + " ; Precision=" + "{0:.2f}%".format(
                    prec[scoreIndex] * 100))
            else:
                print(image_text + "\t||\tscore_threhold=0.5 : F1=0.00% ; Recall=0.00% ; Precision=0.00%")
        apDictionary[class_name] = ap

        n_images = counter_images_per_class.get(class_name)
        log_average_miss_rate_result, miss_rate_result, false_positives_image_result = LogAveMissRate(np.array(rec),
                                                                                    np.array(false_positives), n_images)
        logAveMissRateDictionary[class_name] = log_average_miss_rate_result

        """
         Draw plot
        """
        if plot:
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

    if animation:
        cv2.destroyAllWindows()

    results_file.write("\n# mAP of all classes\n")
    mAP = sumAp / n_classes
    results_text = "mAP = {0:.2f}%".format(mAP * 100)
    results_file.write(results_text + "\n")
    print(results_text)

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

dr_classes = list(det_counter_per_class.keys())

"""
 Plot the total number of occurences of each class in the ground-truth
"""
if plot:
    windowTitle = "ground-truth-info"
    plotTitle = "ground-truth\n"
    plotTitle += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
    xLabel = "Number of objects per class"
    output_path = results_files_path + "/ground-truth-info.png"
    plotShow = False
    plotColor = 'forestgreen'
    draw_plot_func(
        ground_truth_counter_per_class,
        n_classes,
        windowTitle,
        plotTitle,
        xLabel,
        output_path,
        plotShow,
        plotColor,
        '',
    )

"""
 Write number of ground-truth objects per class to results.txt
"""

with os.fdopen(os.open(results_files_path + "/results.txt", OSFLAGS, MODES), 'w') as results_file:
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


"""
 Plot the total number of occurences of each class in the "detection-results" folder
"""
if plot:
    windowTitle = "detection-results-info"
    # Plot title
    plotTitle = "detection-results\n"
    plotTitle += "(" + str(len(dr_files_list)) + " files and "
    count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
    plotTitle += str(count_non_zero_values_in_dictionary) + " detected classes)"
    # end Plot title
    xLabel = "Number of objects per class"
    outputPath = results_files_path + "/detection-results-info.png"
    plotShow = False
    plotColor = 'forestgreen'
    draw_plot_true_p_bar = count_true_positives
    draw_plot_func(
        det_counter_per_class,
        len(det_counter_per_class),
        windowTitle,
        plotTitle,
        xLabel,
        outputPath,
        plotShow,
        plotColor,
        draw_plot_true_p_bar
    )

"""
 Write number of detected objects per class to results.txt
"""
with os.fdopen(os.open(results_files_path + "/results.txt", OSFLAGS, MODES), 'w') as results_file:
    results_file.write("\n# Number of detected objects per class\n")
    for class_name in sorted(dr_classes):
        n_det = det_counter_per_class.get(class_name)
        resule_text = class_name + ": " + str(n_det)
        resule_text += " (tp:" + str(count_true_positives.get(class_name)) + ""
        resule_text += ", false_positives:" + str(n_det - count_true_positives.get(class_name)) + ")\n"
        results_file.write(resule_text)

"""
 Draw log-average miss rate plot (Show logAveMissRate of all classes in decreasing order)
"""
if plot:
    windowTitle = "log_average_miss_rate"
    plotTitle = "log-average miss rate"
    xLabel = "log-average miss rate"
    output_path = results_files_path + "/log_ave_miss_rate.png"
    plotShow = False
    plotColor = 'royalblue'
    draw_plot_func(
        logAveMissRateDictionary,
        n_classes,
        windowTitle,
        plotTitle,
        xLabel,
        output_path,
        plotShow,
        plotColor,
        ""
    )

"""
 Draw mAP plot (Show AP's of all classes in decreasing order)
"""
if plot:
    windowTitle = "mAP"
    plotTitle = "mAP = {0:.2f}%".format(mAP * 100)
    xLabel = "Average Precision"
    output_path = results_files_path + "/mAP.png"
    plotShow = True
    plotColor = 'royalblue'
    draw_plot_func(
        apDictionary,
        n_classes,
        windowTitle,
        plotTitle,
        xLabel,
        output_path,
        plotShow,
        plotColor,
        ""
    )
