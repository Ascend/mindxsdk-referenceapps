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
        cv2.putText(pic, '%d' % (txt), point, fontFace, 0.5, (255, 0, 0))
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
    if img_cv.shape[0] > 8192 or img_cv.shape[1] > 8192 or img_cv.shape[0] < 32 or img_cv.shape[1] < 32:
        print("The test test_image is out of range of between 32 and 8192.")
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
    infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)

    if infer_result.size() == 0:
        print("infer_result is null")
        exit()

    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            infer_result[0].errorCode, infer_result[0].data.decode()))
        exit()

    # Receive results
    visionList = MxpiDataType.MxpiVisionList()
    visionList.ParseFromString(infer_result[0].messageBuf)
    visionData = visionList.visionVec[0].visionData.dataStr
    visionInfo = visionList.visionVec[0].visionInfo

    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    img_yuv = np.frombuffer(visionData, dtype=np.uint8)
    img_yuv = img_yuv.reshape(visionInfo.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo.widthAligned)
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_NV12)
    cv2.imwrite("./source.jpg", img)

    mxpiObjectList = MxpiDataType.MxpiObjectList()
    mxpiObjectList.ParseFromString(infer_result[1].messageBuf)
    print(mxpiObjectList)
    y0 = mxpiObjectList.objectVec[0].y0
    x0 = mxpiObjectList.objectVec[0].x0
    y1 = mxpiObjectList.objectVec[0].y1
    x1 = mxpiObjectList.objectVec[0].x1

    tensorList = MxpiDataType.MxpiTensorPackageList()
    tensorList.ParseFromString(infer_result[2].messageBuf)

    visionListCrop = MxpiDataType.MxpiVisionList()
    visionListCrop.ParseFromString(infer_result[3].messageBuf)
    visionDataCrop = visionListCrop.visionVec[0].visionData.dataStr
    visionInfoCrop = visionListCrop.visionVec[0].visionInfo

    img_yuv_crop = np.frombuffer(visionDataCrop, dtype=np.uint8)
    img_yuv_crop = img_yuv_crop.reshape(visionInfoCrop.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE,
                                        visionInfoCrop.widthAligned)
    img_crop = cv2.cvtColor(img_yuv_crop, cv2.COLOR_YUV2BGR_NV12)

    ids = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
    print(ids)
    result = []
    for j in ids:
        if j >= 0:
            result.append(1)
        elif j < 0:
            result.append(0)
    print(result)

    # Draw detection bounding box
    bboxes = []
    bboxes = {'x0': int(x0),
              'x1': int(x1),
              'y0': int(y0),
              'y1': int(y1),
              'confidence': round(mxpiObjectList.objectVec[0].classVec[0].confidence, 4),
              'text': mxpiObjectList.objectVec[0].classVec[0].className}

    text = "{}{}".format(str(bboxes['confidence']), " ")
    img_copy = copy.copy(img)
    for item in bboxes['text']:
        text += item
    cv2.rectangle(img, (bboxes['x0'], bboxes['y0']), (bboxes['x1'], bboxes['y1']), (255, 0, 0), 2)

    cv2.imwrite("./result.jpg", img)
    cv2.imwrite("./result_crop.jpg", img_crop)

    # Visual attribute recognition results
    index = []
    for i, key in enumerate(result):
        if key == 1:
            index.append(i)

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

    # Classify 35 pedestrian attributes
    text_result = {}
    for i, key in enumerate(index):
        text_result.update({atts[key]: name_dict[atts[key]][0]})
    line = ""
    text_carrying = ""
    text_upper = ""
    text_lower = ""
    text_accessory = ""
    text_foot = ""
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
            text_lower + "\n" + text_accessory + "\n" + text_foot + "\n" + "confidence: " + str(bboxes['confidence'])

    img2 = cv2.imread(img_path)
    cv2.rectangle(img2, (bboxes['x0'], bboxes['y0']), (bboxes['x1'], bboxes['y1']), (255, 0, 0), 2)
    img1 = np.zeros((img2.shape[0], 250, 3), np.uint8)
    img1 = img1 * 0 + 255
    point = (10, 10)
    img1 = write_line(img1, point, line)
    img_rst = np.hstack([img2, img1])
    cv2.imwrite("final_result.jpg", img_rst)

    # destroy streams
    streamManagerApi.DestroyAllStreams()
