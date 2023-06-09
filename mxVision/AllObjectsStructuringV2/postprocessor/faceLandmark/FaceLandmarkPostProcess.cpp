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

#include <cmath>
#include <algorithm>
#include "FaceLandmarkPostProcess.h"

namespace {
    const float ELEMENT_POSITION_LIMIT = 48.0;
    const uint32_t HEAT_MAP_HEIGHT = 5;
    const uint32_t HEAT_MAP_WIDTH = 48 * 48;
    const uint32_t NET_OUTPUT_SHAPE = 2;
    const uint32_t NET_LAYERZ_SIZE = 5;
    const uint32_t NET_INDEX_MAP_WIDTH = 3;
    const int LANDMARK_INFO = 15;
    const uint16_t DEGREE90 = 90;
}

FaceLandmarkPostProcess::FaceLandmarkPostProcess() {}

APP_ERROR FaceLandmarkPostProcess::Init()
{
    LogDebug << "Start to Init FaceLandmarkPostProcess";
    LogDebug << "End to Init FaceLandmarkPostProcess";
    return APP_ERR_OK;
}

APP_ERROR FaceLandmarkPostProcess::DeInit()
{
    return APP_ERR_OK;
}


APP_ERROR FaceLandmarkPostProcess::Process(std::vector<MxBase::Tensor>& inferOutputs,
                                           KeyPointAndAngle& keyPointAndAngle)
{
    LogDebug << "Start to Process FaceLandmarkPostProcess";
    APP_ERROR ret = APP_ERR_OK;
    if (inferOutputs.empty())
    {
        LogError << "result Infer failed with empty output...";
        return APP_ERR_INVALID_PARAM;
    }
   
    float *eulerPtr = static_cast<float *>(inferOutputs[0].GetData());
    float *heatmapPtr = static_cast<float *>(inferOutputs[1].GetData());

    uint8_t indexEulerPtr = 0;
    keyPointAndAngle.angleYaw = fabs(eulerPtr[indexEulerPtr++]) * DEGREE90;
    keyPointAndAngle.anglePitch = fabs(eulerPtr[indexEulerPtr++]) * DEGREE90;
    keyPointAndAngle.angleRoll = fabs(eulerPtr[indexEulerPtr++]) * DEGREE90;

    for (size_t i = 0; i < HEAT_MAP_HEIGHT; i++) {
        float *tempPtr = heatmapPtr + i * HEAT_MAP_WIDTH;
        int position = std::max_element(tempPtr, tempPtr + HEAT_MAP_WIDTH) - tempPtr;
        float x = static_cast<float>((position % static_cast<int32_t>(ELEMENT_POSITION_LIMIT)) / ELEMENT_POSITION_LIMIT);
        float y = static_cast<float>((position / ELEMENT_POSITION_LIMIT) / ELEMENT_POSITION_LIMIT);
        keyPointAndAngle.keyPoints.push_back(x);
        keyPointAndAngle.keyPoints.push_back(y);
    }

    for (size_t i = 0; i < HEAT_MAP_HEIGHT; i++) {
        float *tempPtr = heatmapPtr + i * HEAT_MAP_WIDTH;
        float tempScore = *std::max_element(tempPtr, tempPtr + HEAT_MAP_WIDTH);
        keyPointAndAngle.keyPoints.push_back(tempScore);
    }
    LogDebug << "End to Process FaceLandmarkPostProcess";
    return ret;
}