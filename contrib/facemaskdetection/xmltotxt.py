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
import os
import xml.etree.ElementTree as ET
import glob


def xml_to_txt(indir, outdir):
    """
    Transfer xml to txt
    :param indir: path of xml files
    :param outdir: path of txt files.
    :return:
    """

    os.chdir(indir)
    annotations = os.listdir(".")
    annotations = glob.glob(str(annotations) + "*.xml")

    for i, file in enumerate(annotations):

        file_save = file.split(".")[0] + ".txt"
        file_txt = os.path.join(outdir, file_save)
        f_w = open(file_txt, "w")

        # actual parsing
        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter("object"):
            current = list()
            name = obj.find("name").text
            xmlbox = obj.find("bndbox")
            xn = xmlbox.find("xmin").text
            xx = xmlbox.find("xmax").text
            yn = xmlbox.find("ymin").text
            yx = xmlbox.find("ymax").text
            label = "{} {} {} {} {}".format(name, xn, yn, xx, yx)
            f_w.write(label)
            f_w.write("\n")


if __name__ == "__main__":
    indir1 = "./testimages/FaceMaskDataset/label"  # xml目录
    outdir1 = "./testimages/FaceMaskDataset/ground_truth"  # txt目录

    xml_to_txt(indir1, outdir1)
