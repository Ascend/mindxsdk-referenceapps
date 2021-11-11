#!/usr/bin/env python
# coding=utf-8

"""
 Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.

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
import copy
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


def draw_text(pic, pot, txt, drawType="custom"):
    """
    :param pic:
    :param pot:
    :param text:
    :param drawType: custom or custom
    :return:
    """
    fontScale = 0.4
    thickness = 7
    text_thickness = 1
    bg_color = (255, 255, 255)
    font_size = 0.5
    font_color = (255, 0, 0)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    if drawType == "custom":
        text_size, baseline = cv2.getTextSize(str(txt), fontFace, fontScale, thickness)
        text_loc = (pot[0], pot[1] + text_size[1])
        cv2.rectangle(pic, (text_loc[0] - 2 // 2, text_loc[1] - 2 - baseline),
                      (text_loc[0] + text_size[0], text_loc[1] + text_size[1]), bg_color, -1)
        # draw score value
        cv2.putText(pic, str(txt), (text_loc[0], text_loc[1] + baseline), fontFace, fontScale,
                    (0, 0, 255), text_thickness, 8)
    elif drawType == "simple":
        cv2.putText(pic, '%d' % (txt), point, fontFace, font_size, font_color)
    return pic


def write_line(pic, pot, text_line, drawType="custom"):
    """
    :param pic:
    :param pot:
    :param text:
    :param drawType: custom or custom
    :return:
    """
    fontScale = 0.4
    thickness = 5
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    text_line = text_line.split("\n")
    text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, thickness)
    for i_, text_ in enumerate(text_line):
        if text_:
            draw_point = [pot[0], pot[1] + (text_size[1] + 2 + baseline) * i_]
            pic = draw_text(pic, draw_point, text_, drawType)
    return pic


if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    # init stream manager
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    pipeline_path = b"pipeline/test.pipeline"
    ret = streamManagerApi.CreateMultipleStreamsFromFile(pipeline_path)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    img_path = "dataset/test_image/test9.jpg"
    # Input anomaly detection
    if os.path.exists(img_path) != 1:
        print("The test test_image does not exist.")
        exit()
    img_cv = cv2.imread(img_path)
    max_size = 8192
    min_size = 32
    if img_cv.shape[0] > max_size or img_cv.shape[1] > max_size or \
        img_cv.shape[0] < min_size or img_cv.shape[1] < min_size:
        print("The test test_image is "
        "out of range of between 32 and 8192.")
        exit()
    if img_path.endswith("png"):
        print("This example does not support PNG format image inferencing for the time being.")
        exit()

    streamName = b"detection"
    inPluginId = 0
    dataInput = MxDataInput()
    with open(img_path, 'rb') as f:
        dataInput.data = f.read()
    ret = streamManagerApi.SendData(streamName, inPluginId, dataInput)
    if ret < 0:
        print("Failed to send data to stream")
        exit()

    keyVec = StringVector()
    keyVec.push_back(b"mxpi_imagedecoder0")
    keyVec.push_back(b"mxpi_distributor0_0")
    keyVec.push_back(b"mxpi_tensorinfer1")
    keyVec.push_back(b"mxpi_imagecrop0")
    infer_result = streamManagerApi.GetResult(streamName, b'appsink0', keyVec)

    if infer_result.errorCode != 0:
        print("GetResult error. errorCode=%d, errorMsg=%s" % (
            infer_result.errorCode, infer_result.bufferOutput.data.decode()))

    if infer_result.metadataVec.size() < 2:
        print("No pedestrians detected")
        exit()

    # receive data
    infer_result1 = infer_result.metadataVec[2]
    infer_result2 = infer_result.metadataVec[3]
    infer_result3 = json.loads(infer_result.bufferOutput.data.decode())

    tensorList = MxpiDataType.MxpiTensorPackageList()
    tensorList.ParseFromString(infer_result1.serializedMetadata)
    print(tensorList)

    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    img_yuv = np.frombuffer(visionData, dtype=np.uint8)
    img_yuv = img_yuv.reshape(visionInfo.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo.widthAligned)
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_NV12)
    cv2.imwrite("./source.jpg", img)

    name_dict = {
        "personalLess30": ['less 30'],
        "personalLess45": ['less 45'],
        "personalLess60": ['less 60'],
        "personalLarger60": ['larger 60'],
        "carryingBackpack": ['carrying backpack'],
        "carryingOther": ['carrying other'],
        "lowerBodyCasual": ['casual'],
        "upperBodyCasual": ['casual'],
        "lowerBodyFormal": ['formal'],
        "upperBodyFormal": ['formal'],
        "accessoryHat": ['Hat'],
        "upperBodyJacket": ['Jacket'],
        "lowerBodyJeans": ['jeans'],
        "footwearLeatherShoes": ['Leather Shoes'],
        "upperBodyLogo": ['logo'],
        "hairLong": ['long hair', 'short hair'],
        "personalMale": ['boy', 'girl'],
        "carryingMessengerBag": ['Messenger Bag'],
        "accessoryMuffler": ['Muffler'],
        "accessoryNothing": ['Nothing'],
        "carryingNothing": ['Nothing'],
        "upperBodyPlaid": ['Plaid'],
        "carryingPlasticBags": ['carrying Plastic Bags'],
        "footwearSandals": ['Sandals'],
        "footwearShoes": ['Shoes'],
        "lowerBodyShorts": ['Shorts'],
        "upperBodyShortSleeve": ['Short Sleeve'],
        "lowerBodyShortSkirt": ['Short Skirt'],
        "footwearSneaker": ['Sneaker'],
        "upperBodyThinStripes": ['Thin Stripes'],
        "accessorySunglasses": ['Sunglasses'],
        "lowerBodyTrousers": ['Trousers'],
        "upperBodyTshirt": ['Tshirt'],
        "upperBodyOther": ['Other'],
        "upperBodyVNeck": ['VNeck']
    }
    atts = ["personalLess30", "personalLess45", "personalLess60", "personalLarger60", "carryingBackpack",
            "carryingOther",
            "lowerBodyCasual", "upperBodyCasual", "lowerBodyFormal", "upperBodyFormal",
            "accessoryHat", "upperBodyJacket", "lowerBodyJeans", "footwearLeatherShoes", "upperBodyLogo", "hairLong",
            "personalMale", "carryingMessengerBag", "accessoryMuffler", "accessoryNothing", "carryingNothing",
            "upperBodyPlaid",
            "carryingPlasticBags", "footwearSandals", "footwearShoes", "lowerBodyShorts", "upperBodyShortSleeve",
            "lowerBodyShortSkirt",
            "footwearSneaker", "upperBodyThinStripes", "accessorySunglasses", "lowerBodyTrousers", "upperBodyTshirt",
            "upperBodyOther", "upperBodyVNeck"]
    line = ""

    for key_meta, value_meta in enumerate(tensorList.tensorPackageVec):
        ids = np.frombuffer(tensorList.tensorPackageVec[key_meta].tensorVec[0].dataStr, dtype=np.float32)
        print(ids)
        result = []
        for j in ids:
            if j >= 0:
                result.append(1)
            elif j < 0:
                result.append(0)
        print(result)

        # Visual attribute recognition results
        index = []
        for i, key in enumerate(result):
            if key == 1:
                index.append(i)

    

        # Classify 35 pedestrian attributes
        text_result = {}
        for i, key in enumerate(index):
            text_result.update({atts[key]: name_dict[atts[key]][0]})
        text_carrying = ""
        text_upper = ""
        text_lower = ""
        text_accessory = ""
        text_foot = ""
        line = line + "Pedestrian " + str(key_meta + 1) + ": " + "\n" 
        if "personalMale" in text_result:
            line += "%s: %s\n" % ("gender", name_dict["personalMale"][0])
        else:
            line += "%s: %s\n" % ("gender", name_dict["personalMale"][1])
        if "personalLess30" in text_result:
            line += "%s: %s\n" % ("age", name_dict["personalLess30"][0])
        elif "personalLess45" in text_result:
            line += "%s: %s\n" % ("age", name_dict["personalLess45"][0])
        elif "personalLess60" in text_result:
            line += "%s: %s\n" % ("age", name_dict["personalLess60"][0])
        elif "personalLarger60" in text_result:
            line += "%s: %s\n" % ("age", name_dict["personalLarger60"][0])
        if "hairLong" in text_result:
            line += "%s: %s\n" % ("hair", name_dict["hairLong"][0])
        else:
            line += "%s: %s\n" % ("hair", name_dict["hairLong"][1])

        for key, value in text_result.items():
            if key.startswith('carrying'):
                text_carrying = text_carrying + value + ","
            elif key.startswith('upper'):
                text_upper = text_upper + value + ","
            elif key.startswith('lower'):
                text_lower = text_lower + value + ","
            elif key.startswith('accessory'):
                text_accessory = text_accessory + value + ","
            elif key.startswith('foot'):
                text_foot = text_foot + value + ","
        text_carrying = "carrying: " + text_carrying[:-1]
        text_upper = "upperBody: " + text_upper[:-1]
        text_lower = "lowerBody: " + text_lower[:-1]
        text_accessory = "accessory: " + text_accessory[:-1]
        text_foot = "foot:" + text_foot[:-1]
        line = line + text_carrying + "\n" + text_upper + "\n" + \
               text_lower + "\n" + text_accessory + "\n" + text_foot + "\n" + "confidence: " + str(
            round(infer_result3['MxpiObject'][key_meta]['classVec'][0]['confidence'], 4)) + "\n"

    img2 = cv2.imread(img_path)
    for i, _ in enumerate(infer_result3['MxpiObject']):
        y0 = infer_result3['MxpiObject'][i]['y0']
        x0 = infer_result3['MxpiObject'][i]['x0']
        y1 = infer_result3['MxpiObject'][i]['y1']
        x1 = infer_result3['MxpiObject'][i]['x1']

        # Draw detection bounding box
        bboxes = []
        bboxes = {'x0': int(x0),
                'x1': int(x1),
                'y0': int(y0),
                'y1': int(y1),
                'confidence': round(infer_result3['MxpiObject'][i]['classVec'][0]['confidence'], 4),
                'text': infer_result3['MxpiObject'][i]['classVec'][0]['className']}

        text = "{}{}".format(str(bboxes['confidence']), " ")
        for item in bboxes['text']:
            text += item
        cv2.rectangle(img2, (bboxes['x0'], bboxes['y0']), (bboxes['x1'], bboxes['y1']), (255, 0, 0), 2)
        cv2.putText(img2, "Pedestrian " + str(i + 1), (bboxes['x0'], bboxes['y0'] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)
    col_num = 250
    channel_num = 3
    x = 10
    y = 10
    img1 = np.zeros((img2.shape[0], col_num, channel_num), np.uint8)
    img1 = img1 * 0 + 255
    point = (x, y)
    img1 = write_line(img1, point, line)
    img_rst = np.hstack([img2, img1])
    cv2.imwrite("final_result.jpg", img_rst)

    # destroy streams
    streamManagerApi.DestroyAllStreams()
