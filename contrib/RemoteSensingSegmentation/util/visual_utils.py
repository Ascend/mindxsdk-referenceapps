#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import matplotlib as mpl
from PIL import Image
from matplotlib import pyplot as plt


def semantic_to_mask(mask, labels):
    """
        Turn Semantic diagram to label diagram
        Args:
            mask: semantic diagram
            labels: 8 kinds of semantic tags
        Returns:
            the maximum probability prediction result after label replacement
    """
    x = np.argmax(mask, axis=1)
    label_codes = np.array(labels)
    x = np.uint16(label_codes[x.astype(np.uint8)])
    return x


def decode_seg_map(label_mask, labels, label_colours):
    """
        The result is mapped to a picture
        Args:
            label_mask: the maximum probability prediction result after label replacement
            labels: 8 kinds of semantic tags
            label_colours: colors corresponding to 8 semantic tags
        Returns:
            the prediction result after color replacement
    """
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for i, label in enumerate(labels):
        r[label_mask == label] = label_colours[i, 0]
        g[label_mask == label] = label_colours[i, 1]
        b[label_mask == label] = label_colours[i, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def final_result_create(final_res_pic, label_colours):
    """
        Result image post-processing
        Args:
            final_res_pic: the prediction result after color replacement
            label_colours: colors corresponding to 8 semantic tags
        Returns:
            null
        Output:
            a result picture in directory result/temp_result/result.jpg
    """

    tmp_rgb = np.array(label_colours) / 255.0
    ic_map = mpl.colors.ListedColormap(tmp_rgb, name='my_color')
    # Normalize the data to an interval of [0, 8]
    norm = mpl.colors.Normalize(vmin=0, vmax=8)
    # the size of result picture setting
    plt.figure(figsize=(5, 4))
    h = plt.imshow(final_res_pic, cmap=ic_map, norm=norm)
    c_bar = plt.colorbar(mappable=h)
    # Return evenly spaced numbers over a specified interval
    v = np.linspace(0, 7, 8)
    # Set the location and properties of the ticks
    c_bar.set_ticks((v + 0.5))
    # Set the label of the ticks
    c_bar.set_ticklabels(['水体', '交通运输', '建筑', '林地', '草地', '耕地', '裸土', '其它'])
    plt.savefig('result/temp_result/result.jpg')
    plt.close()


def enable_contrast_output(arr):
    """
        Enable comparison graph output
        Args:
            arr: arr[0] is the img one, arr[1] is the img two, arr[2] is the output directory of the comparison graph result
        Returns:
            null
        Output:
            a comparison graph result in arr[2]
    """
    img1 = Image.open(arr[0])
    img12 = Image.open(arr[1])
    # create a new image, set the width and height
    toImage = Image.new('RGB', (img1.width + img12.width + 35, img12.height), 'white')
    # paste image1 to the new image, and set the position
    toImage.paste(img1, (35, 80))
    # paste image2 to the new image, and set the position
    toImage.paste(img12, (img1.width + 35, 0))
    # save the result image, and the quality is 100
    toImage.save(arr[2], quality=100)
