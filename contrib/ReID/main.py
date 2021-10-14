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
import torch

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
        raise AssertionError(errorMessage)

    # creating stream based on json strings in the pipeline file: 'ReID.pipeline'
    with open("pipeline/ReID.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipelineString = pipeline

    streamState = streamApi.CreateMultipleStreams(pipelineString)
    if streamState != 0:
        errorMessage = "Failed to create Stream, streamState=%s" % str(streamState)
        raise AssertionError(errorMessage)

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
        raise AssertionError(errorMessage)
    if len(os.listdir(queryPath)) == 0:
        errorMessage = 'The query file is empty.'
        raise AssertionError(errorMessage)

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
                    raise AssertionError(errorMessage)

                # get infer result
                inferResult = streamApi.GetProtobuf(QUERY_STREAM_NAME, IN_PLUGIN_ID, pluginNameVector)

                # checking whether the infer results is valid or not
                if inferResult.size() == 0:
                    errorMessage = 'unable to get effective infer results, please check the stream log for details'
                    raise IndexError(errorMessage)
                if inferResult[0].errorCode != 0:
                    errorMessage = "GetProtobuf error. errorCode=%d, errorMessage=%s" % (inferResult[0].errorCode,
                                                                                         inferResult[0].messageName)
                    raise AssertionError(errorMessage)

                # get the output tensor, change it into a numpy array and append it into queryFeatures
                tensorPackage = MxpiDataType.MxpiTensorPackageList()
                tensorPackage.ParseFromString(inferResult[0].messageBuf)
                featureFromTensor = np.frombuffer(tensorPackage.tensorPackageVec[0].tensorVec[0].dataStr,
                                                  dtype=np.float32)
                queryFeatures.append(torch.from_numpy(featureFromTensor))

    # feature reshape and normalization
    queryFeatures = torch.cat(queryFeatures, dim=0)
    queryFeatures = queryFeatures.reshape(FEATURE_RESHAPE_ROW, FEATURE_RESHAPE_COLUMN)
    queryFeatures = torch.nn.functional.normalize(queryFeatures, dim=1, p=2)

    return queryFeatures, queryPid


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
    # constructing the results returned by the galleryImageProcess stream
    pluginNames = [b"mxpi_objectpostprocessor0", b"mxpi_tensorinfer1"]
    pluginNameVector = StringVector()
    for key in pluginNames:
        pluginNameVector.push_back(key)

    # check the gallery file
    if os.path.exists(galleryPath) != 1:
        errorMessage = 'The gallery file does not exist.'
        raise AssertionError(errorMessage)
    if len(os.listdir(galleryPath)) == 0:
        errorMessage = 'The gallery file is empty.'
        raise AssertionError(errorMessage)

    # detect and crop all person for all images in query file, and then extract the features
    for root, dirs, files in os.walk(galleryPath):
        for file in files:
            if file.endswith('.jpg'):
                galleryDataInput = MxDataInput()
                filePath = os.path.join(root, file)
                with open(filePath, 'rb') as f:
                    galleryDataInput.data = f.read()

                # send the prepared data to the stream
                uniqueId = streamApi.SendData(GALLERY_STREAM_NAME, IN_PLUGIN_ID, galleryDataInput)
                if uniqueId < 0:
                    errorMessage = 'Failed to send data to galleryImageProcess stream.'
                    raise AssertionError(errorMessage)

                # get infer result
                inferResult = streamApi.GetProtobuf(GALLERY_STREAM_NAME, IN_PLUGIN_ID, pluginNameVector)

                # checking whether the infer results is valid or not
                if inferResult.size() == 0:
                    errorMessage = 'unable to get effective infer results, please check the stream log for details'
                    raise IndexError(errorMessage)
                if inferResult[0].errorCode != 0:
                    errorMessage = "GetProtobuf error. errorCode=%d, errorMessage=%s" % (inferResult[0].errorCode,
                                                                                         inferResult[0].messageName)
                    raise AssertionError(errorMessage)

                # get the output information of "mxpi_objectpostprocessor0" plugin
                objectList = MxpiDataType.MxpiObjectList()
                objectList.ParseFromString(inferResult[0].messageBuf)

                # get the output information of "mxpi_tensorinfer1" plugin
                featureList = MxpiDataType.MxpiTensorPackageList()
                featureList.ParseFromString(inferResult[1].messageBuf)

                # store the information and features for detected person
                detectedPersonInformation = []
                detectedPersonFeature = []

                # select the detected person, and store its location and features
                for detectedItemIndex in range(0, len(objectList.objectVec)):
                    detectedItem = objectList.objectVec[detectedItemIndex]
                    if detectedItem.classVec[0].className == "person":
                        xLength = int(detectedItem.x1) - int(detectedItem.x0)
                        yLength = int(detectedItem.y1) - int(detectedItem.y0)
                        # ignore the detected person with small size
                        # you can change the threshold
                        if xLength * yLength < DETECTED_PERSON_THRESHOLD:
                            continue
                        detectedPersonInformation.append({'x0': int(detectedItem.x0), 'x1': int(detectedItem.x1),
                                                          'y0': int(detectedItem.y0), 'y1': int(detectedItem.y1)})
                        detectedFeature = \
                            np.frombuffer(featureList.tensorPackageVec[detectedItemIndex].tensorVec[0].dataStr,
                                          dtype=np.float32)
                        detectedPersonFeature.append(torch.from_numpy(detectedFeature))

                # feature reshape and normalization
                detectedPersonFeature = torch.cat(detectedPersonFeature, dim=0)
                detectedPersonFeature = detectedPersonFeature.reshape(FEATURE_RESHAPE_ROW, FEATURE_RESHAPE_COLUMN)
                detectedPersonFeature = torch.nn.functional.normalize(detectedPersonFeature, dim=1, p=2)

                # get the number of the query images
                queryFeatureLength = queryFeatures.shape[0]
                # get the number of the detected persons in this gallery image
                galleryFeatureLength = detectedPersonFeature.shape[0]

                # compute the distance between query feature and gallery feature
                distanceMatrix = torch.pow(queryFeatures, 2).sum(dim=1, keepdim=True).expand(queryFeatureLength,
                                                                                             galleryFeatureLength) + \
                          torch.pow(detectedPersonFeature, 2).sum(dim=1, keepdim=True).expand(galleryFeatureLength,
                                                                                              queryFeatureLength).t()
                distanceMatrix.addmm_(queryFeatures, detectedPersonFeature.t(), beta=1, alpha=-2)
                distanceMatrix = distanceMatrix.cpu().numpy()

                minDistanceIndexMatrix = distanceMatrix.argmin(axis=1)
                minDistanceMatrix = distanceMatrix.min(axis=1)

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

                # read the original image and label the detection results
                image = cv2.imread(filePath)
                # set default legendText and color
                color = NONE_FIND_COLOR

                for galleryIndex in range(0, galleryFeatureLength):
                    # get the locations of the detected person in gallery image
                    locations = detectedPersonInformation[galleryIndex]
                    # if some pid meets the constraints, change the legendText and color
                    if galleryLabelSet[galleryIndex] == 'None':
                        color = NONE_FIND_COLOR
                    else:
                        color = FIND_COLOR
                    # label the detected person in the original image
                    cv2.rectangle(image,
                                  (locations.get('x0'), locations.get('y0')),
                                  (locations.get('x1'), locations.get('y1')),
                                  color, LINE_THICKNESS)
                    cv2.putText(image, galleryLabelSet[galleryIndex],
                                (locations.get('x0'), locations.get('y0')),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, LINE_THICKNESS)
                cv2.imwrite("./result/result_{}".format(str(file)), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queryFilePath', type=str, default='data/querySet', help="Query File Path")
    parser.add_argument('--galleryFilePath', type=str, default='data/gallerySet', help="Gallery File Path")
    parser.add_argument('--matchThreshold', type=float, default=DEFAULT_MATCH_THRESHOLD, help="Match Threshold for ReID Processing")
    opt = parser.parse_args()
    print(opt)
    streamManagerApi = initialize_stream()
    queryFeatureVector, queryPidVector = extract_query_feature(opt.queryFilePath, streamManagerApi)
    process_reid(opt.galleryFilePath, queryFeatureVector, queryPidVector, streamManagerApi, opt.matchThreshold)
