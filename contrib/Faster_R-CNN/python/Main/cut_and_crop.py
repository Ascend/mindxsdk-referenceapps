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


from PIL import Image
import xml.dom.minidom
import os
import cv2
import xml.etree.ElementTree as ET


def cut(img_path, anno_path, cut_path):
    imagelist = os.listdir(img_path)
    annolist = os.listdir(anno_path)
    for image in imagelist:
        image_pre, ext = os.path.splitext(image)
        img_file = img_path + image
        img = Image.open(img_file)
        xmlfile = anno_path + image_pre + ".xml"
        DOMTree = xml.dom.minidom.parse(xmlfile)
        collection = DOMTree.documentElement
        objects = collection.getElementsByTagName("object")
        for object in objects:
            ObjName = object.getElementsByTagName('name')[0].childNodes[0].data
            if ObjName == "hanfeng":
                bndbox = object.getElementsByTagName('bndbox')[0]
                xmin = bndbox.getElementsByTagName('xmin')[0]
                xmin_data = xmin.childNodes[0].data
                ymin = bndbox.getElementsByTagName('ymin')[0]
                ymin_data = ymin.childNodes[0].data
                xmax = bndbox.getElementsByTagName('xmax')[0]
                xmax_data = xmax.childNodes[0].data
                ymax = bndbox.getElementsByTagName('ymax')[0]
                ymax_data = ymax.childNodes[0].data
                if int(ymax_data) - int(ymin_data) > int(xmax_data) - int(xmin_data):
                    if int(xmax_data) - int(xmin_data) > 600:
                        img_cut = img.crop((int(xmin_data), int(ymin_data), int(xmax_data), int(ymax_data)))
                    else:
                        xmin_d = ((int(xmin_data) + int(xmax_data)) / 2 - 300)
                        xmax_d = ((int(xmin_data) + int(xmax_data)) / 2 + 300)
                        img_cut = img.crop((int(xmin_d), int(ymin_data), int(xmax_d), int(ymax_data)))
                else:
                    if int(ymax_data) - int(ymin_data) > 600:
                        ymin_d = ((int(ymin_data) + int(ymax_data)) / 2 - 300)
                        ymax_d = ((int(ymin_data) + int(ymax_data)) / 2 + 300)
                        img_cut = img.crop((int(xmin_data), int(ymin_d), int(xmax_data), int(ymax_d)))
                    else:
                        ymin_d = ((int(ymin_data) + int(ymax_data)) / 2 - 300)
                        ymax_d = ((int(ymin_data) + int(ymax_data)) / 2 + 300)
                        print(ymin_data, ymax_data)
                        img_cut = img.crop((int(xmin_data), int(ymin_d), int(xmax_data), int(ymax_d)))
                img_cut.save(cut_path + image_pre + '.jpg')


def crop_on_slide(cut_path, crop_img_path, stride):
    if not os.path.exists(crop_img_path):
        os.mkdir(crop_img_path)

    output_shape = 600
    imgs = os.listdir(cut_path)

    for img in imgs:
        origin_image = cv2.imread(os.path.join(cut_path, img))
        height = origin_image.shape[0]
        width = origin_image.shape[1]
        print(height)
        print(width)
        x = 0
        newheight = output_shape
        newwidth = output_shape

        while x < width:
            y = 0
            if x + newwidth <= width:
                while y < height:
                    if y + newheight <= height:
                        hmin = y
                        hmax = y + newheight
                        wmin = x
                        wmax = x + newwidth
                    else:
                        hmin = height - newheight
                        hmax = height
                        wmin = x
                        wmax = x + newwidth
                        y = height  # test

                    cropImg1 = os.path.join(crop_img_path, (
                            img.split('.')[0] + '_' + str(wmax) + '_' + str(hmax) + '_' + str(output_shape) + '.jpg'))
                    cv2.imwrite(cropImg1, origin_image[hmin: hmax, wmin: wmax])
                    y = y + stride
                    if y + output_shape == height:
                        y = height
            else:
                while y < height:
                    if y + newheight <= height:
                        hmin = y
                        hmax = y + newheight
                        wmin = width - newwidth
                        wmax = width
                    else:
                        hmin = height - newheight
                        hmax = height
                        wmin = width - newwidth
                        wmax = width
                        y = height  # test

                    cropImg1 = os.path.join(crop_img_path, (
                            img.split('.')[0] + '_' + str(wmax) + '_' + str(hmax) + '_' + str(
                        output_shape) + '.jpg'))
                    cv2.imwrite(cropImg1, origin_image[hmin: hmax, wmin: wmax])
                    y = y + stride
                x = width
            x = x + stride
            if x + output_shape == width:
                x = width


if __name__ == '__main__':
    IMG_PATH = "../data/test/origin_img/"
    XML_PATH = "../data/test/origin_xml/"
    CUT_PATH = "../data/test/cut/"
    cut(IMG_PATH, XML_PATH, CUT_PATH)

    CUT_PATH = "../data/test/cut/"
    crop_img_path = "../data/test/crop/"
    stride = 450
    crop_on_slide(CUT_PATH, crop_img_path, stride)
