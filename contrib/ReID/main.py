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
import cv2
import numpy as np
import io
from PIL import Image

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


GALLERY_STREAM_NAME = b'galleryImageProcess'
QUERY_STREAM_NAME = b'queryImageProcess'

IN_PLUGIN_ID = 0

DETECTED_PERSON_THRESHOLD = 5000
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


def initialize_stream():
    """
    Initialize two streams:
        queryImageProcess for extracting the features of query images
        galleryImageProcess for detecting and re-identifying persons in galley images

    :arg:
        None

    :return:
        Stream api
    """
    streamApi = StreamManagerApi()
    streamState = streamApi.InitManager()
    if streamState != 0:
        errorMessage = "Failed to init Stream manager, streamState=%s" % str(streamState)
        print(errorMessage)
        exit()

    # creating stream based on json strings in the pipeline file: 'ReID.pipeline'
    with open("pipeline/ReID.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipelineString = pipeline

    streamState = streamApi.CreateMultipleStreams(pipelineString)
    if streamState != 0:
        errorMessage = "Failed to create Stream, streamState=%s" % str(streamState)
        print(errorMessage)
        exit()

    return streamApi


def extract_query_feature(queryPath, streamApi):
    """
    Extract the features of query images, return the feature vector and the corresponding Pid vector


    :arg:
        queryPath: the directory of query images
        streamApi: stream api

    :return:
        queryFeatures: the vectors of queryFeatures
        queryPid: the vectors of queryPid
    """
    queryFeatures = []
    queryPid = []

    # constructing the results returned by the queryImageProcess stream
    pluginNames = [b"mxpi_tensorinfer0"]
    pluginNameVector = StringVector()
    for key in pluginNames:
        pluginNameVector.push_back(key)

    # check the query file
    if os.path.exists(queryPath) != 1:
        errorMessage = 'The query file does not exist.'
        print(errorMessage)
        exit()
    if len(os.listdir(queryPath)) == 0:
        errorMessage = 'The query file is empty.'
        print(errorMessage)
        exit()

    # extract the features for all images in query file
    for root, dirs, files in os.walk(queryPath):
        for file in files:
            if file.endswith('.jpg'):
                # store the corresponding pid
                # we use the market1501 as dataset, which is named by
                # 0001(person id)_c1(camera id)s1(sequence id)_000151(frame id)_00(box id).jpg
                # if you use other dataset, modify it to identify the person label
                queryPid.append(file[:4])

                queryDataInput = MxDataInput()
                filePath = os.path.join(root, file)
                with open(filePath, 'rb') as f:
                    queryDataInput.data = f.read()

                # send the prepared data to the stream
                uniqueId = streamApi.SendData(QUERY_STREAM_NAME, IN_PLUGIN_ID, queryDataInput)
                if uniqueId < 0:
                    errorMessage = 'Failed to send data to queryImageProcess stream.'
                    print(errorMessage)
                    exit()

                # get infer result
                inferResult = streamApi.GetProtobuf(QUERY_STREAM_NAME, IN_PLUGIN_ID, pluginNameVector)

                # checking whether the infer results is valid or not
                if inferResult.size() == 0:
                    errorMessage = 'unable to get effective infer results, please check the stream log for details'
                    print(errorMessage)
                    exit()
                if inferResult[0].errorCode != 0:
                    errorMessage = "GetProtobuf error. errorCode=%d, errorMessage=%s" % (inferResult[0].errorCode,
                                                                                         inferResult[0].messageName)
                    print(errorMessage)
                    exit()

                # get the output tensor, change it into a numpy array and append it into queryFeatures
                tensorPackage = MxpiDataType.MxpiTensorPackageList()
                tensorPackage.ParseFromString(inferResult[0].messageBuf)
                featureFromTensor = np.frombuffer(tensorPackage.tensorPackageVec[0].tensorVec[0].dataStr,
                                                  dtype=np.float32)
                cv2.normalize(src=featureFromTensor, dst=featureFromTensor, norm_type=cv2.NORM_L2)
                queryFeatures.append(featureFromTensor.tolist())
            else:
                print('input image only support jpg')
                exit()

    queryFeatures = np.array(queryFeatures)
    return queryFeatures, queryPid


def get_pipeline_results(filePath, streamApi):
    """
    Get the results of current gallery image in pipeline

    :arg:
        filePath: directory of current gallery image
        streamApi: stream api

    :return:
        objectList: results from mxpi_objectpostprocessor0
        featureList: results from mxpi_tensorinfer1
    """
    # constructing the results returned by the galleryImageProcess stream
    pluginNames = [b"mxpi_objectpostprocessor0", b"mxpi_tensorinfer1"]
    pluginNameVector = StringVector()
    for key in pluginNames:
        pluginNameVector.push_back(key)

    galleryDataInput = MxDataInput()
    try:
        image = Image.open(filePath)
        if image.format != 'JPEG':
            print('input image only support jpg')
            exit()
        elif image.width < MIN_IMAGE_SIZE or image.width > MAX_IMAGE_SIZE:
            print('input image width must in range [32, 8192], curr is {}'.format(image.width))
            exit()
        elif image.height < MIN_IMAGE_SIZE or image.height > MAX_IMAGE_SIZE:
            print('input image height must in range [32, 8192], curr is {}'.format(image.height))
            exit()
        else:
            # read input image bytes
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='JPEG')
            galleryDataInput.data = image_bytes.getvalue()
    except IOError:
        print('an IOError occurred while opening {}, maybe your input is not a picture'.format(filePath))
        exit()

    # send the prepared data to the stream
    uniqueId = streamApi.SendData(GALLERY_STREAM_NAME, IN_PLUGIN_ID, galleryDataInput)
    if uniqueId < 0:
        errorMessage = 'Failed to send data to galleryImageProcess stream.'
        print(errorMessage)
        exit()

    # get infer result
    inferResult = streamApi.GetProtobuf(GALLERY_STREAM_NAME, IN_PLUGIN_ID, pluginNameVector)

    # checking whether the infer results is valid or not
    if inferResult.size() == 0:
        errorMessage = 'unable to get effective infer results, please check the stream log for details'
        print(errorMessage)
        exit()
    if inferResult[0].errorCode != 0:
        errorMessage = "GetProtobuf error. errorCode=%d, errorMessage=%s" % (inferResult[0].errorCode,
                                                                             inferResult[0].messageName)
        print(errorMessage)
        exit()

    # get the output information of "mxpi_objectpostprocessor0" plugin
    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(inferResult[0].messageBuf)

    # get the output information of "mxpi_tensorinfer1" plugin
    featureList = MxpiDataType.MxpiTensorPackageList()
    featureList.ParseFromString(inferResult[1].messageBuf)

    return objectList, featureList


def compute_feature_distance(objectList, featureList, queryFeatures):
    """
    Record the location and features of the person in gallery image
    Compute the feature distance between persons in gallery image and query image

    :arg:
        objectList: the results from mxpi_objectpostprocessor0
        featureList: the results from mxpi_tensorinfer1
        queryFeatures: the vectors of queryFeatures

    :return:
        detectedPersonInformation: location information of the detected person in gallery image
        detectedPersonFeature: feature of the detected person in gallery image
        galleryFeatureLength: the length of gallery feature set
        queryFeatureLength: the length of query feature set
        minDistanceIndexMatrix: the index of minimal distance in distance matrix
        minDistanceMatrix: the index of minimal distance value in distance matrix
    """
    # store the information and features for detected person
    detectedPersonInformation = []
    detectedPersonFeature = []
    filterImageCount = 0

    # select the detected person, and store its location and features
    for detectedItemIndex in range(0, len(objectList.objectVec)):
        detectedItem = objectList.objectVec[detectedItemIndex]
        xLength = int(detectedItem.x1) - int(detectedItem.x0)
        yLength = int(detectedItem.y1) - int(detectedItem.y0)
        if xLength < MIN_IMAGE_SIZE or yLength < MIN_IMAGE_WIDTH:
            filterImageCount += 1
            continue
        if detectedItem.classVec[0].className == "person":
            # ignore the detected person with small size
            # you can change the threshold
            if xLength * yLength < DETECTED_PERSON_THRESHOLD:
                continue
            detectedPersonInformation.append({'x0': int(detectedItem.x0), 'x1': int(detectedItem.x1),
                                              'y0': int(detectedItem.y0), 'y1': int(detectedItem.y1)})
            detectedFeature = \
                np.frombuffer(featureList.tensorPackageVec[detectedItemIndex - filterImageCount].tensorVec[0].dataStr,
                              dtype=np.float32)
            cv2.normalize(src=detectedFeature, dst=detectedFeature, norm_type=cv2.NORM_L2)
            detectedPersonFeature.append(detectedFeature.tolist())

    detectedPersonFeature = np.array(detectedPersonFeature)

    # get the number of the query images
    queryFeatureLength = queryFeatures.shape[0]
    # get the number of the detected persons in this gallery image
    galleryFeatureLength = detectedPersonFeature.shape[0]

    # compute the distance between query feature and gallery feature
    distanceMatrix = np.tile(np.sum(np.power(queryFeatureVector, 2), axis=1, keepdims=True),
                             reps=galleryFeatureLength) + \
                     np.tile(np.sum(np.power(detectedPersonFeature, 2), axis=1, keepdims=True),
                             reps=queryFeatureLength).T
    distanceMatrix = ADMM_BETA * distanceMatrix + \
                     ADMM_ALPHA * np.dot(queryFeatureVector, detectedPersonFeature.T)

    # find minimal distance for each query image
    minDistanceIndexMatrix = distanceMatrix.argmin(axis=1)
    minDistanceMatrix = distanceMatrix.min(axis=1)

    return {'detectedPersonInformation': detectedPersonInformation,
            'galleryFeatureLength': galleryFeatureLength, 'queryFeatureLength': queryFeatureLength,
            'minDistanceIndexMatrix': minDistanceIndexMatrix, 'minDistanceMatrix': minDistanceMatrix}


def label_for_gallery_image(galleryFeatureLength, queryFeatureLength, queryPid, minDistanceIndexMatrix,
                            minDistanceMatrix, matchThreshold):
    """
    Label each detected person in gallery image, find the most possible Pid

    :arg:
        galleryFeatureLength: the length of gallery feature set
        queryFeatureLength: the length of query feature set
        queryPid: the vectors of queryPid
        minDistanceIndexMatrix: the index of minimal distance in distance matrix
        minDistanceMatrix: the index of minimal distance value in distance matrix
        matchThreshold: match threshold

    :return:
        galleryLabelSet: labels for current gallery image
    """
    # one person only exists once in each gallery image, thus the Pid in this galleryLabelSet must be unique
    galleryLabelSet = np.full(shape=galleryFeatureLength, fill_value='None')
    galleryLabelDistance = np.full(shape=galleryFeatureLength, fill_value=INITIAL_MIN_DISTANCE, dtype=float)

    for queryIndex in range(0, queryFeatureLength):
        currentPid = queryPid[queryIndex]
        preferGalleryIndex = minDistanceIndexMatrix[queryIndex]
        preferDistance = minDistanceMatrix[queryIndex]
        if preferDistance < matchThreshold:
            pidExistSet = np.where(galleryLabelSet == currentPid)
            pidExistIndex = pidExistSet[0]
            if len(pidExistIndex) == 0:
                if galleryLabelSet[preferGalleryIndex] == 'None':
                    galleryLabelSet[preferGalleryIndex] = currentPid
                    galleryLabelDistance[preferGalleryIndex] = preferDistance
                else:
                    if preferDistance < galleryLabelDistance[preferGalleryIndex]:
                        galleryLabelSet[preferGalleryIndex] = currentPid
                        galleryLabelDistance[preferGalleryIndex] = preferDistance
            else:
                if preferDistance < galleryLabelDistance[pidExistIndex]:
                    galleryLabelSet[pidExistIndex] = 'None'
                    galleryLabelDistance[pidExistIndex] = INITIAL_MIN_DISTANCE
                    galleryLabelSet[preferGalleryIndex] = currentPid
                    galleryLabelDistance[preferGalleryIndex] = preferDistance
    return galleryLabelSet


def draw_results(filePath, galleryFeatureLength, detectedPersonInformation, galleryLabelSet, file):
    """
    Draw and label the detection and re-identification results

    :arg:
        filePath: directory of current gallery image
        galleryFeatureLength: the length of gallery feature set
        detectedPersonInformation: location information of the detected person in gallery image
        galleryLabelSet: labels for current gallery image
        file: name of current gallery image

    :return:
        None
    """
    # read the original image and label the detection results
    image = cv2.imread(filePath)

    for galleryIndex in range(0, galleryFeatureLength):
        # get the locations of the detected person in gallery image
        locations = detectedPersonInformation[galleryIndex]
        # if some pid meets the constraints, change the legendText and color
        if galleryLabelSet[galleryIndex] == 'None':
            color = NONE_FIND_COLOR
        else:
            color = FIND_COLOR
        # label the detected person in the original image
        cv2.rectangle(image, (locations.get('x0'), locations.get('y0')),
                      (locations.get('x1'), locations.get('y1')), color, LINE_THICKNESS)
        cv2.putText(image, galleryLabelSet[galleryIndex], (locations.get('x0'), locations.get('y0')),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, LINE_THICKNESS)
    cv2.imwrite("./result/result_{}".format(str(file)), image)


def process_reid(galleryPath, queryFeatures, queryPid, streamApi, matchThreshold):
    """
    Detect and re-identify person in gallery image

    :arg:
        galleryPath: the directory of gallery images
        queryFeatures: the vectors of queryFeatures
        queryPid: the vectors of queryPid
        streamApi: stream api
        matchThreshold: match threshold

    :return:
        None
    """
    # check the gallery file
    if os.path.exists(galleryPath) != 1:
        errorMessage = 'The gallery file does not exist.'
        print(errorMessage)
        exit()
    if len(os.listdir(galleryPath)) == 0:
        errorMessage = 'The gallery file is empty.'
        print(errorMessage)
        exit()
    outputPath = 'result'
    if os.path.exists(outputPath) != 1:
        errorMessage = 'The result file does not exist.'
        print(errorMessage)
        exit()

    # detect and crop all person for all images in query file, and then extract the features
    for root, dirs, files in os.walk(galleryPath):
        for file in files:
            if file.endswith('.jpg'):
                filePath = os.path.join(root, file)

                objectList, featureList = get_pipeline_results(filePath, streamApi)

                metricDirectory = compute_feature_distance(objectList, featureList, queryFeatures)

                detectedPersonInformation = metricDirectory.get('detectedPersonInformation')
                galleryFeatureLength = metricDirectory.get('galleryFeatureLength')
                queryFeatureLength = metricDirectory.get('queryFeatureLength')
                minDistanceIndexMatrix = metricDirectory.get('minDistanceIndexMatrix')
                minDistanceMatrix = metricDirectory.get('minDistanceMatrix')

                galleryLabelSet = label_for_gallery_image(galleryFeatureLength, queryFeatureLength, queryPid,
                                                          minDistanceIndexMatrix, minDistanceMatrix, matchThreshold)

                draw_results(filePath, galleryFeatureLength, detectedPersonInformation, galleryLabelSet, file)
            else:
                print('input image only support jpg')
                exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queryFilePath', type=str, default='data/querySet', help="Query File Path")
    parser.add_argument('--galleryFilePath', type=str, default='data/gallerySet', help="Gallery File Path")
    parser.add_argument('--matchThreshold', type=float, default=DEFAULT_MATCH_THRESHOLD,
                        help="Match Threshold for ReID Processing")
    opt = parser.parse_args()
    streamManagerApi = initialize_stream()
    queryFeatureVector, queryPidVector = extract_query_feature(opt.queryFilePath, streamManagerApi)
    process_reid(opt.galleryFilePath, queryFeatureVector, queryPidVector, streamManagerApi, opt.matchThreshold)
