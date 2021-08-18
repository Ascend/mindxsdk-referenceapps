"""
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
"""
import os
import argparse
import xml.etree.ElementTree as ET

CLASSES = ('person', 'hat')

def main(arg):
    """
    gain ground-truth
    """
    info = open('./VOC2028.info', 'w')
    cls = open('./voc.names', 'w')
    for i in CLASSES:
        cls.write(i)
        cls.write('\n')

    for idx, jpg_name in enumerate(os.listdir(arg.img_path)):
        key_name = jpg_name.split('.')[0]
        xml_name = os.path.join(arg.ann_path, key_name + '.xml')
        #parse xml
        tree = ET.parse(xml_name)
        root = tree.getroot()
        size = root.find('size')
        width = size.find('width').text
        height = size.find('height').text
        info.write('{} {} {} {}'.format(idx, os.path.join(arg.img_path, jpg_name), width, height))
        info.write('\n')

        with open('{}/{}'.format(arg.gtp, key_name + '.txt'), 'w') as f:
            for obj in root.iter('object'):
                difficult = int(obj.find('difficult').text)
                cls_name = obj.find('name').text.strip().lower()
                if cls_name not in CLASSES:
                    continue
                xml_box = obj.find('bndbox')
                xmin = (float(xml_box.find('xmin').text))
                ymin = (float(xml_box.find('ymin').text))
                xmax = (float(xml_box.find('xmax').text))
                ymax = (float(xml_box.find('ymax').text))

                if difficult:
                    comment = '{} {} {} {} {} {}'.format(cls_name, xmin, ymin, xmax, ymax, 'difficult')
                else:
                    comment = '{} {} {} {} {}'.format(cls_name, xmin, ymin, xmax, ymax)
                f.write(comment)
                f.write('\n')
 

def err_msg(msg):
    """
    print error message
    """
    print('-' * 55)
    print("The specified '{}' file does not exist".format(msg))
    print('You can get the correct parameter information from -h')
    print('-' * 55)
    exit()


def check_args(param):
    """
    check input args
    """
    if not os.path.exists(param.img_path):
        err_msg(param.img_path)
    if not os.path.exists(param.ann_path):
        err_msg(param.ann_path)
    if not os.path.exists(param.gtp):
        os.makedirs(param.gtp)
    return param

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse the VOC2028 dataset label')
    parser.add_argument("--img_path", default="TestImages", help='The image path')
    parser.add_argument("--ann_path", default="Annotations", help='Origin xml path')
    parser.add_argument("--gtp", default="ground-truth/", help='The ground true file path')
    args = parser.parse_args()
    args = check_args(args)
    main(args)