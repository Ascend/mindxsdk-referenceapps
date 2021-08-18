# Copyright 2021 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

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
import numpy as np

MINOVERLAP = 0.5  # default value (defined in the PASCAL VOC2012 challenge)
top_margin = 0.15  # in percentage of the figure height
bottom_margin = 0.05  # in percentage of the figure height
parser = argparse.ArgumentParser()
parser.add_argument('-na', '--no-animation',
                    help="no animation is shown.", action="store_true")
parser.add_argument('-np', '--no-plot',
                    help="no plot is shown.", action="store_true")
parser.add_argument(
    '-q', '--quiet', help="minimalistic console output.", action="store_true")
parser.add_argument('-i', '--ignore', nargs='+', type=str,
                    help="ignore a list of classes.")
parser.add_argument('--set-class-iou', nargs='+', type=str,
                    help="set IoU for a specific class.")

parser.add_argument('--label_path', default="./ground-truth")
parser.add_argument('--npu_txt_path', default="./detection-results")

args = parser.parse_args()


# if there are no classes to ignore then replace None by empty list
if args.ignore is None:
    args.ignore = []

specific_iou_flagged = False
if args.set_class_iou is not None:
    specific_iou_flagged = True

# make sure that the cwd() is the location of
# the python script (so that every path makes sense)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

GT_PATH = args.label_path
DR_PATH = args.npu_txt_path
# if there are no images then no animation can be shown
IMG_PATH = os.path.join(os.getcwd(), 'TestImages_Pre')
if os.path.exists(IMG_PATH):
    for dirpath, dirnames, files in os.walk(IMG_PATH):
        if not files:
            # no image files found
            args.no_animation = True
else:
    args.no_animation = True

# try to import OpenCV if the user didn't choose the option --no-animation
show_animation = False
if not args.no_animation:
    try:
        import cv2

        show_animation = True
    except ImportError:
        print("\"cv2\" not found, please install to visualize.")
        args.no_animation = True

# try to import Matplotlib
# if the user didn't choose the option --no-plot
draw_plot = False
if not args.no_plot:
    try:
        import matplotlib.pyplot as plt

        draw_plot = True
    except ImportError:
        print("\"matplotlib\" not found,install it to get the plots.")
        args.no_plot = True


def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
    log-average miss rate:
        Calculated by averaging miss rates at 9 evenly spaced FPPI points
        between 10e-2 and 10e0, in log-space.
    output:
        lamr | log-average miss rate
        mr | miss rate
        fppi | false positives per image

    references:
     "Pedestrian Detection: An Evaluation of the State of the Art."
    """

    # if there were no detections of that class
    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi

def error(msg):
    """
    throw error and exit
    """
    print(msg)
    sys.exit(0)


def is_float_between_0_and_1(value):
    """
    check if the number is a float between 0.0 and 1.0
    """
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False

def voc_ap(rec, prec):
    """
    Calculate the AP given the recall and precision array
    1) We compute a version of the measured
    precision/recall curve with precision monotonically decreasing
    2) We compute the AP as the area
    under this curve by numerical integration.
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    #  This part makes the precision monotonically decreasing
    #     (goes from the end to the beginning)
    #     matlab: for i=numel(mpre)-1:-1:1
    #                 mpre(i)=max(mpre(i),mpre(i+1));

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    #  This part creates a list of indexes where the recall changes
    #     matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    #  The Average Precision (AP) is the area under the curve
    #     (numerical integration)
    #     matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre

def file_lines_to_list(path):
    """
    Convert the lines of a file to a list
    """
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def draw_text_in_image(img, text0, pos, color, line_width):
    """
    Draws text in image
    """
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text0,
                bottomLeftCornerOfText,
                font,
                fontScale,
                color,
                lineType)
    text_width, _ = cv2.getTextSize(text0, font, fontScale, lineType)[0]
    return img, (line_width + text_width)


def adjust_axes(r, t, fig0, axes0):
    """
    Plot - adjust axes
    """
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig0.dpi
    # get axis width in inches
    current_fig_width = fig0.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes0.get_xlim()
    axes0.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(dictionary0, n_classes0, window_title0, plot_title0, x_label0, output_path0, to_show0,
                   plot_color0, true_p_bar0):
    """
    Draw plot using Matplotlib
    """                
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary0.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar0 != "":
        #  Special case to draw in:
        #     - green -> TP: True Positives
        #     (object detected and matches ground-truth)
        #     - red -> FP: False Positives
        #     (object detected but does not match ground-truth)
        #     - pink -> FN: False Negatives
        #     (object not detected but present in the ground-truth)

        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary0[key] - true_p_bar0[key])
            tp_sorted.append(true_p_bar0[key])
        plt.barh(range(n_classes0), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes0), tp_sorted, align='center', color='forestgreen', label='True Positive',
                 left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        #  Write number on side of bar
        fig = plt.gcf()  # gcf - get current figure
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, plt.gca())
    else:
        plt.barh(range(n_classes0), sorted_values, color=plot_color0)
        #  Write number on side of bar
        fig = plt.gcf()  # gcf - get current figure
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)  # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color0, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, plt.gca())
    # set window title
    fig.canvas.set_window_title(window_title0)
    # write classes in y axis
    plt.yticks(range(n_classes0), sorted_keys, fontsize=12)  # Re-scale height accordingly
    # comput the matrix height in points and inches
    height_pt = n_classes0 * (12 * 1.4)  # 1.4 (some spacing)
    height_in = height_pt / fig.dpi
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height, init_height = fig.get_figheight
    if figure_height > fig.get_figheight():
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title0, fontsize=14)
    # set axis titles
    plt.xlabel(x_label0, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path0)
    # show image
    if to_show0:
        plt.show()
    # close the plot
    plt.close()


#  Create a ".temp_files/" and "output/" directory
TEMP_FILES_PATH = ".temp_files"
if not os.path.exists(TEMP_FILES_PATH):  # if it doesn't exist already
    os.makedirs(TEMP_FILES_PATH)
output_files_path = "output"
if os.path.exists(output_files_path):  # if it exist already
    # reset the output directory
    shutil.rmtree(output_files_path)

os.makedirs(output_files_path)
if draw_plot:
    os.makedirs(os.path.join(output_files_path, "classes"))
if show_animation:
    os.makedirs(os.path.join(output_files_path,
                             "images", "detections_one_by_one"))

#  ground-truth
#      Load each of the ground-truth files
#      into a temporary ".json" file.
#      Create a list of all the class names present
#      in the ground-truth (gt_classes).
# get a list with the ground-truth files
ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
if len(ground_truth_files_list) == 0:
    error("Error: No ground-truth files found!")
ground_truth_files_list.sort()
# dictionary with counter per class
gt_counter_per_class = {}
counter_images_per_class = {}

gt_files = []
for txt_file in ground_truth_files_list:
    file_id = txt_file.split(".txt", 1)[0]

    file_id = os.path.basename(os.path.normpath(file_id))
    # check if there is a correspondent detection-results file
    temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
    if not os.path.exists(temp_path):
        # error_msg = "Error. File not found: {}\n".format(temp_path)
        # error(error_msg)
        continue
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
        except ValueError:
            error_msg = "Error: File " + txt_file + " in the wrong format.\n"
            error_msg += " Expected: <class_name> <l> <t> <r> <b>\n"
            error_msg += " Received: " + line
            error(error_msg)
        # check if class is in the ignore list, if yes skip
        if class_name == "hat":
            class_name = "helmet"
        elif class_name == "person":
            class_name = "head"
        if class_name in args.ignore:
            continue
        bbox = left + " " + top + " " + right + " " + bottom
        if is_difficult:
            bounding_boxes.append(
                {"class_name": class_name, "bbox": bbox,
                 "used": False, "difficult": True})
            is_difficult = False
        else:
            bounding_boxes.append(
                {"class_name": class_name, "bbox": bbox, "used": False})
            # count that object
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1

            if class_name not in already_seen_classes:
                if class_name in counter_images_per_class:
                    counter_images_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    counter_images_per_class[class_name] = 1
                already_seen_classes.append(class_name)

    # dump bounding_boxes into a ".json" file
    new_temp_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
    gt_files.append(new_temp_file)
    with open(new_temp_file, 'w') as outfile:
        json.dump(bounding_boxes, outfile)


gt_classes = list(gt_counter_per_class.keys())
print(gt_counter_per_class)
# let's sort the classes alphabetically
gt_classes = sorted(gt_classes)
n_classes = len(gt_classes)

# """
#  Check format of the flag --set-class-iou (if used)
#     e.g. check if class exists
# """
if specific_iou_flagged:
    n_args = len(args.set_class_iou)
    error_msg = \
        '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
    if n_args % 2 != 0:
        error('Error, missing arguments. Flag usage:' + error_msg)
    # [class_1] [IoU_1] [class_2] [IoU_2]
    specific_iou_classes = args.set_class_iou[::2]  # even
    iou_list = args.set_class_iou[1::2]  # odd
    if len(specific_iou_classes) != len(iou_list):
        error('Error, missing arguments. Flag usage:' + error_msg)
    for tmp_class in specific_iou_classes:
        if tmp_class not in gt_classes:
            error('Error, unknown class \"' + tmp_class +
                  '\". Flag usage:' + error_msg)
    for num in iou_list:
        if not is_float_between_0_and_1(num):
            error('IoU must be [0.0,1.0].usage:' + error_msg)

# """
#  detection-results
#      Load each of the detection-results files
#      into a temporary ".json" file.
# """
# get a list with the detection-results files
dr_files_list = glob.glob(DR_PATH + '/*.txt')
# print(dr_files_list)
dr_files_list.sort()

for class_index, class_name in enumerate(gt_classes):
    bounding_boxes = []
    for txt_file in dr_files_list:
        # the first time it checks
        # if all the corresponding ground-truth files exist
        file_id = txt_file.split(".txt", 1)[0]
        # print("file_id",file_id)

        file_id = os.path.basename(os.path.normpath(file_id))
        temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
        if class_index == 0:
            if not os.path.exists(temp_path):
                error_msg = "Error. File not found: {}\n".format(temp_path)
                error(error_msg)
        lines = file_lines_to_list(txt_file)
        for line in lines:
            try:
                sl = line.split()
                tmp_class_name, confidence, left, top, right, bottom = sl
            except ValueError:
                error_msg = "Error: File " + txt_file + " wrong format.\n"
                error_msg += " Expected: <classname> <conf> <l> <t> <r> <b>\n"
                error_msg += " Received: " + line
                error(error_msg)
            if tmp_class_name == class_name:
                bbox = left + " " + top + " " + right + " " + bottom
                bounding_boxes.append(
                    {"confidence": confidence,
                     "file_id": file_id, "bbox": bbox})

    # sort detection-results by decreasing confidence
    bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
    with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
        json.dump(bounding_boxes, outfile)

# """
#  Calculate the AP for each class
# """
sum_AP = 0.0
ap_dictionary = {}
lamr_dictionary = {}
# open file to store the output
with open(output_files_path + "/output.txt", 'w') as output_file:
    output_file.write("# AP and precision/recall per class\n")
    count_true_positives = {}
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        # """
        #  Load detection-results of that class
        # """
        dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
        dr_data = json.load(open(dr_file))

        # """
        #  Assign detection-results to ground-truth objects
        # """
        nd = len(dr_data)
        tp = [0] * nd  # creates an array of zeros of size nd
        fp = [0] * nd
        count = 0
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            if show_animation:
                # find ground truth image
                ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
                if len(ground_truth_img) == 0:
                    error("Error. Image not found with id: " + file_id)
                elif len(ground_truth_img) > 1:
                    error("Error. Multiple image with id: " + file_id)
                else:  # found image
                    # Load image
                    img = cv2.imread(IMG_PATH + "/" + ground_truth_img[0])
                    # load image with draws of multiple detections
                    img_cumulative_path = output_files_path
                    img_cumulative_path += "/images/" + ground_truth_img[0]
                    if os.path.isfile(img_cumulative_path):
                        img_cumulative = cv2.imread(img_cumulative_path)
                    else:
                        img_cumulative = img.copy()
                    # Add bottom border to image
                    bottom_border = 60
                    BLACK = [0, 0, 0]
                    img = cv2.copyMakeBorder(
                        img, 0, bottom_border,
                        0, 0, cv2.BORDER_CONSTANT, value=BLACK)
            # assign detection-results to ground truth object if any
            # open ground-truth with that file_id
            gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = [float(x) for x in detection["bbox"].split()]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]),
                          min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU)
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
                             (bbgt[2] - bbgt[0] + 1) * \
                             (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            # assign detection as true positive/don't care/false positive
            if show_animation:
                status = "NO MATCH FOUND!"
            # set minimum overlap
            min_overlap = MINOVERLAP
            if specific_iou_flagged:
                if class_name in specific_iou_classes:
                    index = specific_iou_classes.index(class_name)
                    min_overlap = float(iou_list[index])
            if ovmax >= min_overlap:
                if "difficult" not in gt_match:
                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        # update the ".json" file
                        with open(gt_file, 'w') as f:
                            f.write(json.dumps(ground_truth_data))
                        if show_animation:
                            status = "MATCH!"
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
                        if show_animation:
                            status = "REPEATED MATCH!"
            else:
                # false positive
                fp[idx] = 1
                count += 1
                if ovmax > 0:
                    status = "INSUFFICIENT OVERLAP"

            # """
            #  Draw image to show animation
            # """
            if show_animation:
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
                img, line_width = draw_text_in_image(
                    img, text, (margin, v_pos), white, 0)
                text = "Class [" + str(class_index) + "/" + \
                       str(n_classes) + "]: " + class_name + " "
                img, line_width = draw_text_in_image(
                    img, text,
                    (margin + line_width, v_pos),
                    light_blue, line_width)
                if ovmax != -1:
                    color = light_red
                    if status == "INSUFFICIENT OVERLAP":
                        text = "IoU: {0:.2f}% ".format(
                            ovmax * 100)
                        text += "< {0:.2f}% ".format(min_overlap * 100)
                    else:
                        text = "IoU: {0:.2f}% ".format(
                            ovmax * 100)
                        text += ">= {0:.2f}% ".format(min_overlap * 100)
                        color = green
                    img, _ = draw_text_in_image(
                        img, text, (margin + line_width, v_pos),
                        color, line_width)
                # 2nd line
                v_pos += int(bottom_border / 2.0)
                rank_pos = str(idx + 1)  # rank position (idx starts at 0)
                text = "Detection #rank: " + rank_pos
                temp_conf = float(detection["confidence"])
                text += " conf: {0:.2f}% ".format(temp_conf * 100)
                img, line_width = draw_text_in_image(
                    img, text, (margin, v_pos), white, 0)
                color = light_red
                if status == "MATCH!":
                    color = green
                text = "Result: " + status + " "
                img, line_width = draw_text_in_image(
                    img, text, (margin + line_width, v_pos), color, line_width)

                font = cv2.FONT_HERSHEY_SIMPLEX
                # if there is intersections between the bounding-boxes
                if ovmax > 0:
                    bbgt = [int(round(float(x)))
                            for x in gt_match["bbox"].split()]
                    cv2.rectangle(img, (bbgt[0], bbgt[1]),
                                  (bbgt[2], bbgt[3]), light_blue, 2)
                    cv2.rectangle(img_cumulative, (bbgt[0], bbgt[1]),
                                  (bbgt[2], bbgt[3]), light_blue, 2)
                    cv2.putText(img_cumulative, class_name,
                                (bbgt[0], bbgt[1] - 5), font, 0.6,
                                light_blue, 1, cv2.LINE_AA)
                bb = [int(i) for i in bb]
                cv2.rectangle(img, (bb[0], bb[1]),
                              (bb[2], bb[3]), color, 2)
                cv2.rectangle(img_cumulative,
                              (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                cv2.putText(img_cumulative, class_name,
                            (bb[0], bb[1] - 5), font,
                            0.6, color, 1, cv2.LINE_AA)
                # show image
                cv2.imwrite("result.jpg", img)
                # save image to output
                output_img_path = output_files_path
                output_img_path += "/images/detections_one_by_one/"
                output_img_path += class_name + "_detection"
                output_img_path += str(idx) + ".jpg"
                cv2.imwrite(output_img_path, img)
                # save the image with all the objects drawn to it
                cv2.imwrite(img_cumulative_path, img_cumulative)

        print("f_count:", count)
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        # class_name + " AP = {0:.2f}%".format(ap*100)
        text = "{0:.2f}%".format(ap * 100) + " = " + class_name + " AP "
        # """
        #  Write to output.txt
        # """
        rounded_prec = ['%.2f' % elem for elem in prec]
        rounded_rec = ['%.2f' % elem for elem in rec]
        output_file.write(text + "\n Precision: " + str(rounded_prec) +
                          "\n Recall :" + str(rounded_rec) + "\n\n")
        if not args.quiet:
            print(text)
        ap_dictionary[class_name] = ap

        n_images = counter_images_per_class[class_name]
        lamr, mr, fppi = log_average_miss_rate(
            np.array(rec), np.array(fp), n_images)
        lamr_dictionary[class_name] = lamr

        # """
        #  Draw plot
        # """
        if draw_plot:
            plt.plot(rec, prec, '-o')
            # add a new penultimate point to the list (mrec[-2], 0.0)
            # since the last line segment
            # (and respective area) do not affect the AP value
            area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
            area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
            plt.fill_between(area_under_curve_x, 0,
                             area_under_curve_y, alpha=0.2, edgecolor='r')
            # set window title
            fig = plt.gcf()  # gcf - get current figure
            fig.canvas.set_window_title('AP ' + class_name)
            # set plot title
            plt.title('class: ' + text)
            # set axis titles
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # optional - set axes
            axes = plt.gca()  # gca - get current axes
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
            # Alternative option -> wait for button to be pressed
            # Alternative option -> normal display
            # save the plot
            fig.savefig(output_files_path + "/classes/" + class_name + ".png")
            plt.cla()  # clear axes for next plot

    if show_animation:
        cv2.destroyAllWindows()

    output_file.write("\n# mAP of all classes\n")
    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP * 100)
    output_file.write(text + "\n")
    print(text)

# """
#  Draw false negatives
# """
if show_animation:
    pink = (203, 192, 255)
    for tmp_file in gt_files:
        ground_truth_data = json.load(open(tmp_file))
        # get name of corresponding image
        start = TEMP_FILES_PATH + '/'
        img_id = tmp_file[tmp_file.find(
            start) + len(start):tmp_file.rfind('_ground_truth.json')]
        img_cumulative_path = output_files_path + "/images/" + img_id + ".jpg"
        img = cv2.imread(img_cumulative_path)
        if img is None:
            img_path = IMG_PATH + '/' + img_id + ".jpg"
            img = cv2.imread(img_path)
        # draw false negatives
        for obj in ground_truth_data:
            if not obj['used']:
                bbgt = [int(round(float(x))) for x in obj["bbox"].split()]
                cv2.rectangle(img, (bbgt[0], bbgt[1]),
                              (bbgt[2], bbgt[3]), pink, 2)
        cv2.imwrite(img_cumulative_path, img)

# remove the temp_files directory
shutil.rmtree(TEMP_FILES_PATH)

# """
#  Count total of detection-results
# """
# iterate through all the files
det_counter_per_class = {}
for txt_file in dr_files_list:
    file_id = txt_file.split(".txt", 1)[0]
    # print("file_id",file_id)

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
print("dr_class:", dr_classes)

# """
#  Plot the total number of occurences of each class in the ground-truth
# """
if draw_plot:
    window_title = "ground-truth-info"
    plot_title = "ground-truth\n"
    plot_title += "(" + str(len(ground_truth_files_list)) + \
                  " files and " + str(n_classes) + " classes)"
    x_label = "Number of objects per class"
    output_path = output_files_path + "/ground-truth-info.png"
    to_show = False
    plot_color = 'forestgreen'
    draw_plot_func(
        gt_counter_per_class,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        '',
    )

# """
#  Write number of ground-truth objects per class to results.txt
# """
with open(output_files_path + "/output.txt", 'a') as output_file:
    output_file.write("\n# Number of ground-truth objects per class\n")
    for class_name in sorted(gt_counter_per_class):
        output_file.write(class_name + ": " +
                          str(gt_counter_per_class[class_name]) + "\n")

# """
# Finish counting true positives
# if class exists in detection-result
# but not in ground-truth
# then there are no true positives in that class
# """
for class_name in dr_classes:
    if class_name not in gt_classes:
        count_true_positives[class_name] = 0
print("count_true_p:", count_true_positives)
# """
#  Plot the total number of occurences of
#  each class in the "detection-results" folder
# """
if draw_plot:
    window_title = "detection-results-info"
    # Plot title
    plot_title = "detection-results\n"
    plot_title += "(" + str(len(dr_files_list)) + " files and "
    count_non_zero_values_in_dictionary = sum(
        int(x) > 0 for x in list(det_counter_per_class.values()))
    plot_title += str(count_non_zero_values_in_dictionary)
    plot_title += " detected classes)"
    # end Plot title
    x_label = "Number of objects per class"
    output_path = output_files_path + "/detection-results-info.png"
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

# """
#  Write number of detected objects per class to output.txt
# """
with open(output_files_path + "/output.txt", 'a') as output_file:
    output_file.write("\n# Number of detected objects per class\n")
    for class_name in sorted(dr_classes):
        n_det = det_counter_per_class[class_name]
        text = class_name + ": " + str(n_det)
        text += " (tp:" + str(count_true_positives[class_name]) + ""
        text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
        output_file.write(text)

# """
#  Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
# """
if draw_plot:
    window_title = "lamr"
    plot_title = "log-average miss rate"
    x_label = "log-average miss rate"
    output_path = output_files_path + "/lamr.png"
    to_show = False
    plot_color = 'royalblue'
    draw_plot_func(
        lamr_dictionary,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
    )

# """
#  Draw mAP plot (Show AP's of all classes in decreasing order)
# """
if draw_plot:
    window_title = "mAP"
    plot_title = "mAP = {0:.2f}%".format(mAP * 100)
    x_label = "Average Precision"
    output_path = output_files_path + "/mAP.png"
    to_show = True
    plot_color = 'royalblue'
    draw_plot_func(
        ap_dictionary,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
    )
