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

#include "PPYoloePostProcess.h"
#include <algorithm>
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"

namespace MxBase {
const int OUTPUT_SIZE = 2;
const int MODEL_BOX_NUM = 8400;
const int RIGHTX_IDX = 2;
const int RIGHTY_IDX = 3;
const int OUTPUT_DIMS = 3;
PPYoloePostProcess::PPYoloePostProcess() {}

PPYoloePostProcess &PPYoloePostProcess::operator = (const PPYoloePostProcess &other)
{
    if (this == &other) {
        return *this;
    }
    ObjectPostProcessBase::operator = (other);
    objectnessThresh_ = other.objectnessThresh_;
    iouThresh_ = other.iouThresh_;
    return *this;
}

APP_ERROR PPYoloePostProcess::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig)
{
    LogDebug << "Start to Init PPYoloePostProcess.";
    APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
        return ret;
    }
    ret = configData_.GetFileValue<float>("OBJECTNESS_THRESH", objectnessThresh_, 0.0f, 1.0f);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "Fail to read OBJECTNESS_THRESH from config, default is : " << objectnessThresh_;
    }
    ret = configData_.GetFileValue<float>("IOU_THRESH", iouThresh_, 0.0f, 1.0f);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "Fail to read IOU_THRESH from config, default is : " << iouThresh_;
    }
    LogDebug << "End to Init PPYoloePostProcess.";
    return APP_ERR_OK;
}

APP_ERROR PPYoloePostProcess::Init(const std::map<std::string, std::string> &postConfig)
{
    LogDebug << "Start to Init PPYoloePostProcess.";
    APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
        return ret;
    }
    ret = configData_.GetFileValue<float>("OBJECTNESS_THRESH", objectnessThresh_, 0.0f, 1.0f);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "Fail to read OBJECTNESS_THRESH from config, default is : " << objectnessThresh_;
    }
    ret = configData_.GetFileValue<float>("IOU_THRESH", iouThresh_, 0.0f, 1.0f);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "Fail to read IOU_THRESH from config, default is : " << iouThresh_;
    }
    LogDebug << "End to Init PPYoloePostProcess.";
    return APP_ERR_OK;
}

APP_ERROR PPYoloePostProcess::DeInit()
{
    return APP_ERR_OK;
}

void PPYoloePostProcess::ConstructBoxFromOutput(float *output, float *boxOutput, size_t offset,
    std::vector<ObjectInfo> &objectInfo, const ResizedImageInfo &resizedImageInfo)
{
    int classId = -1;
    float maxProb = scoreThresh_;
    for (int labelIdx = 0; labelIdx < classNum_; labelIdx++) {
        if (maxProb < output[offset + labelIdx * MODEL_BOX_NUM]) {
            maxProb = output[offset + labelIdx * MODEL_BOX_NUM];
            classId = labelIdx;
        }
    }
    if (classId < 0 || resizedImageInfo.widthOriginal == 0 || resizedImageInfo.heightOriginal == 0) {
        return;
    }
    float xGain = resizedImageInfo.widthResize * 1.0 / resizedImageInfo.widthOriginal;
    float yGain = resizedImageInfo.heightResize * 1.0 / resizedImageInfo.heightOriginal;
    auto leftX = boxOutput[offset * BOX_DIM] / xGain;
    auto leftY = (boxOutput[offset * BOX_DIM + 1]) / yGain;
    auto rightX = (boxOutput[offset * BOX_DIM + RIGHTX_IDX]) / xGain;
    auto rightY = (boxOutput[offset * BOX_DIM + RIGHTY_IDX]) / yGain;
    ObjectInfo obj;
    obj.x0 = leftX;
    obj.y0 = leftY;
    obj.x1 = rightX;
    obj.y1 = rightY;
    obj.confidence = maxProb;
    obj.classId = classId;
    obj.className = configData_.GetClassName(obj.classId);
    if (maxProb < separateScoreThresh_[obj.classId])
        return;
    objectInfo.push_back(obj);
}

void PPYoloePostProcess::LogObjectInfo(std::vector<std::vector<ObjectInfo>> &objectInfos)
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

APP_ERROR PPYoloePostProcess::Process(const std::vector<TensorBase> &tensors,
    std::vector<std::vector<ObjectInfo>> &objectInfos, const std::vector<ResizedImageInfo> &resizedImageInfos,
    const std::map<std::string, std::shared_ptr<void>> &paramMap)
{
    LogDebug << "Start to Process PPYoloePostProcess.";
    if (resizedImageInfos.empty() || tensors.size() < OUTPUT_SIZE || tensors[1].GetShape().size() != OUTPUT_DIMS) {
        LogError << "Tensors or ResizedImageInfos is not provided for ppyoloe postprocess";
        return APP_ERR_INPUT_NOT_MATCH;
    }
    if (tensors[1].GetShape()[1] != classNum_) {
        LogError << "The model output tensor[1][1] != classNum_.";
        return APP_ERR_INPUT_NOT_MATCH;
    }
    uint32_t batchSize = tensors[0].GetShape()[0];
    if (resizedImageInfos.size() != batchSize) {
        LogError << "The size of resizedImageInfo does not match the batchSize of tensors.";
        return APP_ERR_INPUT_NOT_MATCH;
    }
    int rows = tensors[1].GetSize() / (classNum_ * batchSize);
    for (size_t k = 0; k < batchSize; k++) {
        auto output = (float*)GetBuffer(tensors[1], k);
        auto boxOutput = (float*)GetBuffer(tensors[0], k);
        std::vector<ObjectInfo> objectInfo;
        for (size_t i = 0; i < rows; ++i) {
            ConstructBoxFromOutput(output, boxOutput, i, objectInfo, resizedImageInfos[k]);
        }
        MxBase::NmsSort(objectInfo, iouThresh_);
        objectInfos.push_back(objectInfo);
    }
    LogObjectInfo(objectInfos);
    LogDebug << "End to Process PPYoloePostProcess.";
    return APP_ERR_OK;
}

extern "C" {
std::shared_ptr<MxBase::PPYoloePostProcess> GetObjectInstance()
{
    LogInfo << "Begin to get PPYoloePostProcess instance.";
    auto instance = std::make_shared<MxBase::PPYoloePostProcess>();
    LogInfo << "End to get PPYoloePostProcess instance.";
    return instance;
}
}
}