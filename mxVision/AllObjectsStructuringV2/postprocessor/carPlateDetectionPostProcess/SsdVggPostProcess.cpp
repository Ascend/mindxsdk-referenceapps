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

#include "SsdVggPostProcess.h"

namespace {
    const uint32_t INFONUM = 8;
    const uint32_t CLASSID = 1;
    const uint32_t CONFIDENCE = 2;
    const uint32_t LEFTTOPX = 3;
    const uint32_t LEFTTOPY = 4;
    const uint32_t RIGHTBOTX = 5;
    const uint32_t RIGHTBOTY = 6;
}

SsdVggPostProcess::SsdVggPostProcess() {}

APP_ERROR SsdVggPostProcess::Init(const std::map<std::string, std::string> &postConfig)
{
    LogDebug << "Start to Init SsdVggPostprocess";
    APP_ERROR ret = APP_ERR_OK;
    std::string postProcessConfigPath;
    std::string lablePath;
    if (postConfig.find("postProcessConfigPath") != postConfig.end() &&
    !(postConfig.find("postProcessConfigPath")->second).empty()) {
        postProcessConfigPath = postConfig.find("postProcessConfigPath")->second;
    }
    if (postConfig.find("lablePath") != postConfig.end() &&
        !(postConfig.find("lablePath")->second).empty()) {
        lablePath = postConfig.find("lablePath")->second;
    }

    ret = util_.LoadConfiguration(postProcessConfigPath, configData_, MxBase::CONFIGFILE);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to get postprocess config";
        return ret;
    }
    ret = configData_.LoadLabels(lablePath);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to get postprocess labels";
        return ret;
    }
    ret = configData_.GetFileValue<uint32_t>("CLASS_NUM", classNum_);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to get CLASS_NUM";
        return ret;
    }
    ret = configData_.GetFileValue<float>("SCORE_THRESH", scoreThresh_);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to get SCORE_THRESH";
        return ret;
    }
    LogDebug << "End to Init SsdVggPostprocess";
    return ret;
}

APP_ERROR SsdVggPostProcess::DeInit()
{
    return APP_ERR_OK;
}

APP_ERROR SsdVggPostProcess::Process(const MxBase::Image& originImage, const std::vector<MxBase::Tensor>& inferOutputs,
                                     std::vector<MxBase::ObjectInfo>& objectInfos)
{
    LogDebug << "Start to Process SsdVggPostprocess";
    APP_ERROR ret = APP_ERR_OK;
    if (inferOutputs.empty())
    {
        LogError << "result Infer failed with empty output...";
        return APP_ERR_INVALID_PARAM;
    }

    int *objectNum = static_cast<int *>(inferOutputs[0].GetData());
    float *objectInfo = static_cast<float *>(inferOutputs[1].GetData());

    for (int i = 0; i < *objectNum; i++) {
        uint32_t classId = static_cast<uint32_t>(objectInfo[i * INFONUM + CLASSID]);
        float confidence = objectInfo[i * INFONUM + CONFIDENCE];
        if (classId >= classNum_ || confidence < scoreThresh_) {
            continue;
        }

        MxBase::ObjectInfo objInfo;
        objInfo.className = configData_.GetClassName(classId);
        objInfo.confidence = confidence;
        objInfo.classId = classId;
        objInfo.x0 = objectInfo[i * INFONUM + LEFTTOPX] * originImage.GetOriginalSize().width;
        objInfo.y0 = objectInfo[i * INFONUM + LEFTTOPY] * originImage.GetOriginalSize().height;
        objInfo.x1 = objectInfo[i * INFONUM + RIGHTBOTX]* originImage.GetOriginalSize().width;
        objInfo.y1 = objectInfo[i * INFONUM + RIGHTBOTY]* originImage.GetOriginalSize().height;
        objectInfos.push_back(objInfo);
    }
    LogDebug << "End to Process SsdVggPostprocess";
    return ret;
}