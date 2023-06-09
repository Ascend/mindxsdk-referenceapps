/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <map>
#include <thread>
#include <chrono>
#include <iostream>
#include <queue>
#include <memory>
#include "unistd.h"

#include "MxBase/Maths/FastMath.h"
#include "MxBase/MxBase.h"
#include "MxBase/postprocess/include/ObjectPostProcessors/Yolov3PostProcess.h"
#include "MxBase/E2eInfer/ImageProcessor/ImageProcessor.h"
#include "MxBase/DeviceManager/DeviceManager.h"

#include "postprocessor/resnetAttributePostProcess/resnetAttributePostProcess.h"
#include "postprocessor/carPlateDetectionPostProcess/SsdVggPostProcess.h"
#include "postprocessor/carPlateRecognitionPostProcess/carPlateRecognitionPostProcess.h"
#include "postprocessor/faceLandmark/FaceLandmarkPostProcess.h"
#include "postprocessor/faceAlignment/FaceAlignment.h"
#include "utils/objectSelection/objectSelection.h"

#include "taskflow/taskflow.hpp"
#include "taskflow/algorithm/pipeline.hpp"
#include "BlockingQueue.h"

std::string CLASSNAMEPERSON = "person";
std::string CLASSNAMEVEHICLE = "motor-vehicle";
std::string CLASSNAMEFACE = "face";

size_t maxQueueSize = 32;
float minQueuePercent = 0.2;
float maxQueuePercent = 0.8;

const size_t numChannel = 80;
const size_t numWoker = 10;
const size_t numLines = 8;

uint32_t deviceID = 0;
std::vector<uint32_t> deviceIDs(numChannel, deviceID);

// yolo detection
MxBase::ImageProcessor *imageProcessors[numWoker];
MxBase::Model *yoloModels[numWoker];
MxBase::Yolov3PostProcess *yoloPostProcessors[numWoker];
MultiObjectTracker *multiObjectTrackers[numWoker];

// vehicle attribution
MxBase::Model *vehicleAttrModels[numWoker];
ResNetAttributePostProcess *vehicleAttrPostProcessors[numWoker];

// car plate detection
MxBase::Model *carPlateDetectModels[numWoker];
SsdVggPostProcess *carPlateDetectPostProcessors[numWoker];

// car plate recognition
MxBase::Model *carPlateRecModels[numWoker];
carPlateRecognitionPostProcess *carPlateRecPostProcessors[numWoker];

// pedestrian attribution
MxBase::Model *pedestrianAttrModels[numWoker];
ResnetAttributePostProcess *pedestrianAttrPostProcessors[numWoker];

// pedestrian feature
MxBase::Model *pedestrianFeatureModels[numWoker];

// face landmarks
MxBase::Model *faceLandmarkModels[numWoker];
FaceLandmarkPostProcess *faceLandmarkPostProcessors[numWoker];

// face alignment
FaceAlignment *faceAlignmentProcessors[numWoker];

// face attribution

// face feature
