/*
 * Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

#include "Yolov7PostProcess.h"
#include <algorithm>
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"
#include "MxBase/DeviceManager/DeviceManager.h"

namespace MxBase {
const int MODEL_INPUT_SIZE = 640;
const int ALIGN_LEFT = 16;
const int CONFIDENCE_IDX = 4;
const int LABEL_START_OFFSET = 5;
const int OUTPUT_DIMS = 3;
const int XOFFSET = 2;
const int YOFFSET = 3;
const int AVG_PARAM = 2;
const float EPSILON = 1e-6;
Yolov7PostProcess::Yolov7PostProcess() {}

Yolov7PostProcess &Yolov7PostProcess::operator = (const Yolov7PostProcess &other)
{
    if (this == &other) {
        return *this;
    }
    ObjectPostProcessBase::operator = (other);
    objectnessThresh_ = other.objectnessThresh_;
    iouThresh_ = other.iouThresh_;
    paddingType_ = other.paddingType_;
    return *this;
}

APP_ERROR Yolov7PostProcess::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig)
{
    LogDebug << "Start to Init Yolov7PostProcess. ";
    APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessorBase";
        return ret;
    }
    ret = configData_.GetFileValue<float>("OBJECTNESS_THRESH", objectnessThresh_, 0.0f, 1.0f);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "Fail to read OBJECTNESS_THRESH from config, default is :" << objectnessThresh_;
    }
    ret = configData_.GetFileValue<float>("IOU_THRESH", iouThresh_, 0.0f, 1.0f);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "Fail to read IOU_THRESH from config, default is :" << iouThresh_;
    }
    ret = configData_.GetFileValue<int>("PADDING_TYPE", paddingType_, 0, 1);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "Fail to read PADDING_TYPE from config, default is :" << paddingType_;
    }
    LogDebug << "End to Init Yolov7PostProcess. ";
    return APP_ERR_OK;
}

APP_ERROR Yolov7PostProcess::Init(const std::map<std::string, std::string> &postConfig)
{
    LogDebug << "Start to Init Yolov7PostProcess. ";
    APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessorBase";
        return ret;
    }
    ret = configData_.GetFileValue<float>("OBJECTNESS_THRESH", objectnessThresh_, 0.0f, 1.0f);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "Fail to read OBJECTNESS_THRESH from config, default is :" << objectnessThresh_;
    }
    ret = configData_.GetFileValue<float>("IOU_THRESH", iouThresh_, 0.0f, 1.0f);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "Fail to read IOU_THRESH from config, default is :" << iouThresh_;
    }
    ret = configData_.GetFileValue<int>("PADDING_TYPE", paddingType_, 0, 1);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "Fail to read PADDING_TYPE from config, default is :" << paddingType_;
    }
    LogDebug << "End to Init Yolov7PostProcess. ";
    return APP_ERR_OK;
}

APP_ERROR Yolov7PostProcess::DeInit()
{
    return APP_ERR_OK;
}

void Yolov7PostProcess::ConstructBoxFromOutput(float *output, size_t offset, std::vector<ObjectInfo> &objectInfo,
    const ResizedImageInfo &resizedImageInfo)
{
    size_t index = offset * (classNum_ + LABEL_START_OFFSET);
    if (output[index + CONFIDENCE_IDX] <= objectnessThresh_) {
        return;
    }
    int classId = -1;
    float maxProb = scoreThresh_;
    for (int j = LABEL_START_OFFSET; j < classNum_ + LABEL_START_OFFSET; j++) {
        if (output[index + j] * output[index + CONFIDENCE_IDX] > maxProb) {
            maxProb = output[index + j] * output[index + CONFIDENCE_IDX];
            classId = j - LABEL_START_OFFSET;
        }
    }
    if (classId < 0) {
        return;
    }
    int tmpResizedWidth = resizedImageInfo.widthResize;
    int tmpResizedHeight = resizedImageInfo.heightResize;
    double division = 1;
    if (std::fabs(resizedImageInfo.keepAspectRatioScaling) > EPSILON) {
        division = resizedImageInfo.keepAspectRatioScaling;
    }
    if (tmpResizedWidth == tmpResizedHeight && tmpResizedHeight == MODEL_INPUT_SIZE) {
        tmpResizedWidth = std::round(resizedImageInfo.widthOriginal * division);
        tmpResizedHeight = std::round(resizedImageInfo.heightOriginal * division);
    }
    int offsetLeft = (MODEL_INPUT_SIZE - tmpResizedWidth) / AVG_PARAM;
    int offsetTop = (MODEL_INPUT_SIZE - tmpResizedHeight) / AVG_PARAM;
    if (paddingType_ == 0) {
        offsetTop = offsetTop % AVG_PARAM ? offsetTop : offsetTop - 1;
        if (DeviceManager::IsAscend310() && resizedImageInfo.resizeType == RESIZER_MS_KEEP_ASPECT_RATIO) {
            offsetLeft = (offsetLeft - 1 + ALIGN_LEFT) / ALIGN_LEFT * ALIGN_LEFT;
        } else {
            offsetLeft = offsetLeft < ALIGN_LEFT ? 0 : offsetLeft / ALIGN_LEFT * ALIGN_LEFT;
        }
    }
    auto leftX = (output[index] - output[index + XOFFSET] / AVG_PARAM - offsetLeft) / division;
    auto leftY = (output[index + 1] - output[index + YOFFSET] / AVG_PARAM - offsetTop) / division;
    auto rightX = (output[index] + output[index + XOFFSET] / AVG_PARAM - offsetLeft) / division;
    auto rightY = (output[index + 1] + output[index + YOFFSET] / AVG_PARAM - offsetTop) / division;

    ObjectInfo obj;
    obj.x0 = leftX < 0.0 ? 0.0 : leftX;
    obj.y0 = leftY < 0.0 ? 0.0 : leftY;
    obj.x1 = rightX > resizedImageInfo.widthOriginal ? resizedImageInfo.widthOriginal : rightX;
    obj.y1 = rightY > resizedImageInfo.heightOriginal ? resizedImageInfo.heightOriginal : rightY;
    obj.confidence = maxProb;
    obj.classId = classId;
    obj.className = configData_.GetClassName(obj.classId);
    if (maxProb < separateScoreThresh_[obj.classId])
        return;
    objectInfo.push_back(obj);
}

void Yolov7PostProcess::LogObjectInfo(std::vector<std::vector<ObjectInfo>> &objectInfos)
{
    for (size_t i = 0; i < objectInfos.size(); i++) {
        LogDebug << "Objects in Image No." << i << " are listed:";
        for (auto &objInfo : objectInfos[i]) {
            auto number = (separateScoreThresh_.size() > objInfo.classId) ? separateScoreThresh_[(int)objInfo.classId] :
                scoreThresh_;
            LogDebug << "Find object: classId(" << objInfo.classId << "), confidence("  << objInfo.confidence <<
                "), scoreThresh(" << number <<"), Coordinates (x0, y0)=(" << objInfo.x0 << ", " << objInfo.y0 <<
                "); (x1, y1)=(" << objInfo.x1 << ", " << objInfo.y1 << ").";
        }
    }
}

APP_ERROR Yolov7PostProcess::Process(const std::vector<TensorBase> &tensors,
    std::vector<std::vector<ObjectInfo>> &objectInfos, const std::vector<ResizedImageInfo> &resizedImageInfos,
    const std::map<std::string, std::shared_ptr<void>> &paramMap)
{
    LogDebug << "Start to Process Yolov7PostProcess.";
    if (resizedImageInfos.empty() || tensors.empty() || tensors[0].GetShape().size() != OUTPUT_DIMS) {
        LogError << "Tensors or ResizedImageInfos is not provided for yolov7 postprocess";
        return APP_ERR_INPUT_NOT_MATCH;
    }
    if (tensors[0].GetShape()[OUTPUT_DIMS - 1] != classNum_ + LABEL_START_OFFSET) {
        LogError << "The model output tensor[2] != classNum_.";
        return APP_ERR_INPUT_NOT_MATCH;
    }
    if (resizedImageInfos.size() != tensors.size()) {
        LogError << "The size of resizedImageInfos does not match the size of tensors.";
        return APP_ERR_INPUT_NOT_MATCH;
    }
    uint32_t batchSize = tensors[0].GetShape()[0];
    size_t rows = tensors[0].GetSize() / ((classNum_ + LABEL_START_OFFSET) * batchSize);
    for (size_t k = 0; k < batchSize; k++) {
        auto output = (float*)GetBuffer(tensors[0], k);
        std::vector<ObjectInfo> objectInfo;
        for (size_t i = 0; i < rows; ++i) {
            ConstructBoxFromOutput(output, i, objectInfo, resizedImageInfos[k]);
        }
        MxBase::NmsSort(objectInfo, iouThresh_);
        objectInfos.push_back(objectInfo);
    }
    LogObjectInfo(objectInfos);
    LogDebug << "End to Process Yolov7PostProcess.";
    return APP_ERR_OK;
}

extern "C" {
std::shared_ptr<MxBase::Yolov7PostProcess> GetObjectInstance()
{
    LogInfo << "Begin to get Yolov7PostProcess instance";
    auto instance = std::make_shared<MxBase::Yolov7PostProcess>();
    LogInfo << "End to get Yolov7PostProcess instance";
    return instance;
}
}
}