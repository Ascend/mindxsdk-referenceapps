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

import argparse
import os
import io
import time
import threading
import multiprocessing
import cv2
import numpy as np

from mindx.sdk import base
from mindx.sdk.base import ImageProcessor
from PIL import Image
from mindx.sdk.base import Tensor, Model, Size, Rect, log, ImageProcessor, post, Point

IN_PLUGIN_ID = 0

INITIAL_MIN_DISTANCE = 99

LINE_THICKNESS = 2
FONT_SCALE = 1.0
FIND_COLOR = (0, 255, 0)
NONE_FIND_COLOR = (255, 0, 0)

DEFAULT_MATCH_THRESHOLD = 0.3
FEATURE_RESHAPE_ROW = -1
FEATURE_RESHAPE_COLUMN = 2048
ADMM_BETA = 1
ADMM_ALPHA = -2

MIN_IMAGE_SIZE = 32
MAX_IMAGE_SIZE = 8192
MIN_IMAGE_WIDTH = 6
DEVICE_ID = 1
NUM_THREADS_OF_INFER = 32
YOLOV3_MODEL_PATH = "./models/yolov3.om"
YOLOV3 = base.model(YOLOV3_MODEL_PATH, deviceId=DEVICE_ID)
REID_MODEL_PATH = "./models/ReID.om"
REID = base.model(REID_MODEL_PATH, deviceId=DEVICE_ID)
LABEL_PATH = "/home/yuanlei2/cjb/mindxsdk-referenceapps/contrib/ReID/models/coco.names"  # 分类标签文件的路�?
CONFIG_PATH = "/home/yuanlei2/cjb/mindxsdk-referenceapps/contrib/ReID/models/yolov3.cfg"

imageProcessor1 = ImageProcessor(DEVICE_ID)
yolov3_post = post.Yolov3PostProcess(config_path=CONFIG_PATH, label_path=LABEL_PATH)


def extract_query_feature(querypath):
    """
    Extract the features of query images, return the feature vector and the corresponding Pid vector


    :arg:
        querypath: the directory of query images
        streamApi: stream api

    :return:
        queryfeatures: the vectors of queryfeatures
        querypid: the vectors of querypid
    """
    queryfeatures = []
    querypid = []
    # constructing the results returned by the queryImageProcess
    # check the query file
    if os.path.exists(querypath) != 1:
        errormessage = 'The query file does not exist.'
        print(errormessage)
        exit()
    if len(os.listdir(querypath)) == 0:
        errormessage = 'The query file is empty.'
        print(errormessage)
        exit()

    # extract the features for all images in query file
    for root, dirs, files in os.walk(querypath):
        for file in files:
            if file.endswith('.jpg'):
                # store the corresponding pid
                # we use the market1501 as dataset, which is named by
                # 0001(person id)_c1(camera id)s1(sequence id)_000151(frame id)_00(box id).jpg
                # if you use other dataset, modify it to identify the person label
                querypid.append(file[:4])
                imageprocessor = ImageProcessor(DEVICE_ID)
                filepath = os.path.join(root, file)
                decodedimg = imageprocessor.decode(filepath, base.nv12)
                size_cof = Size(128, 256)
                resizedimg = imageprocessor.resize(decodedimg, size_cof, base.huaweiu_high_order_filter)
                imgtensor = [resizedimg.to_tensor()]
                reid_output = REID.infer(imgtensor)
                reid_output0 = reid_output[0]
                reid_output0.to_host()
                queryfeature = np.array(reid_output0)
                cv2.normalize(src=queryfeature, dst=queryfeature, norm_type=cv2.NORM_L2)
                queryfeatures.append(queryfeature)

            else:
                print('Input image only support jpg')
                exit()
    return queryfeatures, querypid


def get_pipeline_results(filepath):
    """
    Get the results of current gallery image in pipeline

    :arg:
        filepath: directory of current gallery image
        streamApi: stream api

    :return:
        objectList: results from mxpi_objectpostprocessor0
        featureList: results from mxpi_tensorinfer1
    """
    # constructing the results returned by the galleryImageProcess stream

    try:
        image = Image.open(filepath)
        if image.format != 'JPEG':
            print('Input image only support jpg')
            exit()
        elif image.width < MIN_IMAGE_SIZE or image.width > MAX_IMAGE_SIZE:
            print('Input image width must in range [32, 8192], curr is {}'.format(image.width))
            exit()
        elif image.height < MIN_IMAGE_SIZE or image.height > MAX_IMAGE_SIZE:
            print('Input image height must in range [32, 8192], curr is {}'.format(image.height))
            exit()
        else:
            # read input image bytes
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='JPEG')
    except IOError:
        print('An IOError occurred while opening {}, maybe your input is not a picture'.format(filepath))
        exit()

    resizeinfo = base.ResizedImageInfo()

    resizeinfo.heightResize = 416
    resizeinfo.widthResize = 416
    resizeinfo.heightOriginal = 1080
    resizeinfo.widthOriginal = 1920

    decodedimg = imageProcessor1.decode(filepath, base.nv12)
    size_cof = Size(416, 416)
    resizedimg = imageProcessor1.resize(decodedimg, size_cof, base.huaweiu_high_order_filter)
    imgtensor1 = [resizedimg.to_tensor()]
    yolov3_outputs = YOLOV3.infer(imgtensor1)

    # 构造后处理的输入
    inputs = []
    len1 = len(yolov3_outputs)
    for x in range(len1):
        yolov3_outputs[x].to_host()
        n = np.array(yolov3_outputs[x])
        tensor = Tensor(n)
        inputs.append(tensor)
    yolov3_post_results = yolov3_post.process(inputs, [resizeinfo])
    cropresizevec = []
    objectlist = []
    len2 = len(yolov3_post_results)
    for i in range(len2):
        for j in range(len(yolov3_post_results[i])):
            x0 = int(yolov3_post_results[i][j].x0)
            y0 = int(yolov3_post_results[i][j].y0)
            x1 = int(yolov3_post_results[i][j].x1)
            y1 = int(yolov3_post_results[i][j].y1)
            classname = yolov3_post_results[i][j].className
            objectlist.append([x0, y0, x1, y1, classname])
            cropresizevec.append((Rect(x0, y0, x1, y1), Size(128, 256)))
    yolov3_crop = imageProcessor1.crop_resize(decodedimg, cropresizevec)
    imgtensor2 = [x.to_tensor() for x in yolov3_crop]
    featurelist = []
    len3 = len(imgtensor2)
    for x in range(len3):
        reid_output = REID.infer([imgtensor2[x]])
        reid_output[0].to_host()
        featurelist.append(np.array(reid_output[0]))

    return objectlist, featurelist


def compute_feature_distance(objectlist, featurelist, queryfeatures):
    """
    Record the location and features of the person in gallery image
    Compute the feature distance between persons in gallery image and query image

    :arg:
        objectlist: the results from mxpi_objectpostprocessor0
        featurelist: the results from mxpi_tensorinfer1
        queryfeatures: the vectors of queryfeatures

    :return:
        detectedpersoninformation: location information of the detected person in gallery image
        detectedpersonfeature: feature of the detected person in gallery image
        galleryFeatureLength: the length of gallery feature set
        queryFeatureLength: the length of query feature set
        minDistanceIndexMatrix: the index of minimal distance in distance matrix
        minDistanceMatrix: the index of minimal distance value in distance matrix
    """
    # store the information and features for detected person
    detectedpersoninformation = []
    detectedpersonfeature = []

    filterimagecount = 0

    persondetectedflag = False

    # select the detected person, and store its location and features
    len1 = len(objectlist)
    for detecteditemindex in range(len1):
        detecteditem = objectlist[detecteditemindex]
        xlength = int(detecteditem[2]) - int(detecteditem[0])
        ylength = int(detecteditem[3]) - int(detecteditem[1])
        if xlength < MIN_IMAGE_SIZE or ylength < MIN_IMAGE_WIDTH:
            filterimagecount += 1
            continue
        if detecteditem[4] == "person":
            persondetectedflag = True
            detectedpersoninformation.append({'x0': int(detecteditem[0]), 'x1': int(detecteditem[2]),
                                              'y0': int(detecteditem[1]), 'y1': int(detecteditem[3])})
            detectedfeature = featurelist[detecteditemindex]
            cv2.normalize(src=detectedfeature, dst=detectedfeature, norm_type=cv2.NORM_L2)
            detectedpersonfeature.append(detectedfeature)

    if not persondetectedflag:
        return None

    # get the number of the query images
    queryfeaturelength = len(queryfeatures)
    queryfeaturevector1 = np.array(queryFeatureVector).reshape(queryfeaturelength, FEATURE_RESHAPE_COLUMN)

    # get the number of the detected persons in this gallery image
    galleryfeaturelength = len(detectedpersonfeature)
    detectedpersonfeature = np.array(detectedpersonfeature).reshape(galleryfeaturelength, FEATURE_RESHAPE_COLUMN)

    # # compute the distance between query feature and gallery feature
    distancematrix = np.tile(np.sum(np.power(queryfeaturevector1, 2), axis=1, keepdims=True),
                             reps=galleryfeaturelength) + \
                     np.tile(np.sum(np.power(detectedpersonfeature, 2), axis=1, keepdims=True),
                             reps=queryfeaturelength).T
    distancematrix = ADMM_BETA * distancematrix + \
                     ADMM_ALPHA * np.dot(queryfeaturevector1, detectedpersonfeature.T)

    # find minimal distance for each query image
    mindistanceindexmatrix = distancematrix.argmin(axis=1)
    mindistancematrix = distancematrix.min(axis=1)

    return {'detectedpersoninformation': detectedpersoninformation,
            'galleryfeaturelength': galleryfeaturelength, 'queryfeaturelength': queryfeaturelength,
            'mindistanceindexmatrix': mindistanceindexmatrix, 'mindistancematrix': mindistancematrix}


def label_for_gallery_image(galleryfeaturelength, queryfeaturelength, querypid, mindistanceindexmatrix,
                            mindistancematrix, matchthreshold):
    """
    Label each detected person in gallery image, find the most possible Pid

    :arg:
        galleryfeaturelength: the length of gallery feature set
        queryfeaturelength: the length of query feature set
        querypid: the vectors of querypid
        mindistanceindexmatrix: the index of minimal distance in distance matrix
        mindistancematrix: the index of minimal distance value in distance matrix
        matchthreshold: match threshold

    :return:
        gallerylabelset: labels for current gallery image
    """
    # one person only exists once in each gallery image, thus the Pid in this gallerylabelset must be unique
    gallerylabelset = np.full(shape=galleryfeaturelength, fill_value='None')
    gallerylabeldistance = np.full(shape=galleryfeaturelength, fill_value=INITIAL_MIN_DISTANCE, dtype=float)

    for queryindex in range(0, queryfeaturelength):
        currentpid = querypid[queryindex]
        prefergalleryindex = mindistanceindexmatrix[queryindex]
        preferdistance = mindistancematrix[queryindex]
        if preferdistance < matchthreshold:
            pidexistset = np.where(gallerylabelset == currentpid)
            pidexistindex = pidexistset[0]
            if len(pidexistindex) == 0:
                if gallerylabelset[prefergalleryindex] == 'None':
                    gallerylabelset[prefergalleryindex] = currentpid
                    gallerylabeldistance[prefergalleryindex] = preferdistance
                else:
                    if preferdistance < gallerylabeldistance[prefergalleryindex]:
                        gallerylabelset[prefergalleryindex] = currentpid
                        gallerylabeldistance[prefergalleryindex] = preferdistance
            else:
                if preferdistance < gallerylabeldistance[pidexistindex]:
                    gallerylabelset[pidexistindex] = 'None'
                    gallerylabeldistance[pidexistindex] = INITIAL_MIN_DISTANCE
                    gallerylabelset[prefergalleryindex] = currentpid
                    gallerylabeldistance[prefergalleryindex] = preferdistance
    return gallerylabelset


def draw_results(filepath, galleryfeaturelength, detectedpersoninformation, gallerylabelset, file):
    """
    Draw and label the detection and re-identification results

    :arg:
        filepath: directory of current gallery image
        galleryfeaturelength: the length of gallery feature set
        detectedpersoninformation: location information of the detected person in gallery image
        gallerylabelset: labels for current gallery image
        file: name of current gallery image

    :return:
        None
    """
    # read the original image and label the detection results
    image = cv2.imread(filepath)

    for galleryindex in range(0, galleryfeaturelength):
        # get the locations of the detected person in gallery image
        locations = detectedpersoninformation[galleryindex]
        # if some pid meets the constraints, change the legendText and color
        if gallerylabelset[galleryindex] == 'None':
            color = NONE_FIND_COLOR
        else:
            color = FIND_COLOR
        # label the detected person in the original image
        cv2.rectangle(image, (locations.get('x0'), locations.get('y0')),
                      (locations.get('x1'), locations.get('y1')), color, LINE_THICKNESS)
        cv2.putText(image, gallerylabelset[galleryindex], (locations.get('x0'), locations.get('y0')),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, LINE_THICKNESS)
    cv2.imwrite("./result/result_{}".format(str(file)), image)
    print("Detect ", file, " successfully.")


class DrawThread(threading.Thread):
    def __init__(self, queue_in, queue_out):
        threading.Thread.__init__(self)
        self.flag = True
        self.queue_in = queue_in
        self.queue_out = queue_out

    def run(self):
        while self.flag:
            if self.queue_in.empty() is False:
                try:
                    inputs = self.queue_in.get(timeout=1)
                except ValueError as e:
                    print(e)
                    continue
                draw_results(*inputs)
                self.queue_out.put(1)


def process_new(gallerypath, queryfeatures, querypid, matchthreshold):
    """
    Detect and re-identify person in gallery image

    :arg:
        gallerypath: the directory of gallery images
        queryfeatures: the vectors of queryfeatures
        querypid: the vectors of querypid
        matchthreshold: match threshold

    :return:
        None
    """

    if os.path.exists(gallerypath) != 1:
        errormessage = 'The gallery file does not exist.'
        print(errormessage)
        exit()
    if len(os.listdir(gallerypath)) == 0:
        errormessage = 'The gallery file is empty.'
        print(errormessage)
        exit()
    outputpath = 'result'
    if os.path.exists(outputpath) != 1:
        errormessage = 'The result file does not exist.'
        print(errormessage)
        exit()

    # 进程间的通信队列
    queue_result = multiprocessing.Manager().Queue()
    queue_count = multiprocessing.Manager().Queue()

    thread2 = DrawThread(queue_result, queue_count)
    thread2.start()

    num_files = 0
    count = 0
    inputs = []
    for root, dirs, files in os.walk(gallerypath):
        for file in files:
            if file.endswith('.jpg'):
                filepath = os.path.join(root, file)
                num_files += 1
                objectlist, featurelist = get_pipeline_results(filepath)
                metricdirectory = compute_feature_distance(objectlist, featurelist, queryfeatures)

                if not metricdirectory:
                    print("Cannot detect person for image:", file)
                    continue

                detectedpersoninformation = metricdirectory.get('detectedpersoninformation')
                galleryfeaturelength = metricdirectory.get('galleryfeaturelength')
                queryfeaturelength = metricdirectory.get('queryfeaturelength')
                mindistanceindexmatrix = metricdirectory.get('mindistanceindexmatrix')
                mindistancematrix = metricdirectory.get('mindistancematrix')

                gallerylabelset = label_for_gallery_image(galleryfeaturelength, queryfeaturelength, querypid,
                                                          mindistanceindexmatrix, mindistancematrix, matchthreshold)

                queue_result.put([filepath, galleryfeaturelength, detectedpersoninformation, gallerylabelset, file])

    while count != num_files:
        if queue_count.empty() is False:
            try:
                inputs = queue_count.get(timeout=1)
            except ValueError as e:
                print(e)
                continue
            count += 1

    if count == num_files:
        thread2.flag = False
        thread2.join()




if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--queryFilePath', type=str, default='data/querySet/', help="Query File Path")
    parser.add_argument('--galleryFilePath', type=str, default='data/gallerySet/', help="Gallery File Path")
    parser.add_argument('--matchThreshold', type=float, default=DEFAULT_MATCH_THRESHOLD,
                        help="Match Threshold for ReID Processing")
    opt = parser.parse_args()
    queryFeatureVector, queryPidVector = extract_query_feature(opt.queryFilePath)
    process_new(opt.galleryFilePath, queryFeatureVector, queryPidVector, opt.matchThreshold)
    end = time.time()
    total_time = float(end - start)
    print('V2 Running time: %f Seconds' % total_time)










