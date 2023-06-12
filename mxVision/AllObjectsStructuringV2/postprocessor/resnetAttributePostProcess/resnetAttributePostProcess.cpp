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

#include "resnetAttributePostProcess.h"

APP_ERROR ResnetAttributePostProcess::GetAttributeIndex(std::string &strAttributeIndex)
{
    if (attributeNum_ <= 0)
    {
        LogError << GetError(APP_ERR_ACL_FAILURE) << "Failed to get attributeNum(" << strAttributeIndex << ").";
    }
    attributionIndex_.clear();
    size_t i = 0;
    size_t num = strAttributeIndex.find('%');
    while (num != std::string::npos && i < attributeNum_)
    {
        std::string attributeIndexGroup = strAttributeIndex.substr(0, num);
        attributionIndex_.emplace_back();
        size_t indexNum = attributeIndexGroup.find(',');
        while (indexNum != std::string::npos)
        {
            std::string oneAttributeIndex = attributeIndexGroup.substr(0, indexNum);
            try
            {
                attributionIndex_[i].emplace_back(stof(oneAttributeIndex));
            }
            catch (std::exception e)
            {
                LogError << "oneAttributeIndex string(" << attributeIndexGroup << ") cast to float failed.";
                return APP_ERR_COMM_INVALID_PARAM;
            }
            indexNum++;
            attributeIndexGroup = attributeIndexGroup.substr(indexNum, attributeIndexGroup.size());
            indexNum = attributeIndexGroup.find(',');
        }

        try
        {
            attributionIndex_[i].emplace_back(stof(attributeIndexGroup));
        }
        catch (std::exception e)
        {
            LogError << "attributeIndexGroup string(" << attributeIndexGroup << ") cast to float failed.";
            return APP_ERR_COMM_INVALID_PARAM;
        }
        num++;
        strAttributeIndex = strAttributeIndex.substr(num, strAttributeIndex.size());
        i++;
        num = strAttributeIndex.find('%');
    }
    return APP_ERR_OK;
}

APP_ERROR ResnetAttributePostProcess::Process(std::vector<MxBase::Tensor> &resnetInferResVec, std::vector<std::vector<ResnetAttr>> &attributeResVec)
{
    std::vector<float> result;
    for (auto &resnetInferRes : resnetInferResVec)
    {
        for (int j = 0; j < resnetInferRes.GetShape()[1]; j++)
        {
            auto *castData = static_cast<float *>(resnetInferRes.GetData() + j * sizeof(float));
            if (activationFunc_ == "sigmoid")
            {
                result.push_back(fastmath::sigmoid(*castData));
            }
            else
            {
                result.push_back(*castData);
            }
        }
    }

    std::vector<ResnetAttr> objectResnetAttr;
    for (int i = 0; i < attributionIndex_.size(); i++)
    {
        ResnetAttr resnetAttr;
        int argmaxIndex = -1;
        float currentMax = 0;
        for (auto index : attributionIndex_[i])
        {
            if (result[index] > currentMax)
            {
                currentMax = result[index];
                argmaxIndex = index;
            }
        }

        std::string attributeValueStr = attributeValueVec_[argmaxIndex];
        if (attributionIndex_[i].size() == 1)
        {
            size_t split = attributeValueStr.find('|');
            std::vector<std::string> attributeValues;
            attributeValues.emplace_back(attributeValueStr.substr(0, split));
            attributeValues.emplace_back(attributeValueStr.substr(split + 1, attributeValueStr.size()));
            if (currentMax >= CONFIDENCE)
            {
                resnetAttr.attrValue = attributeValues[0];
                resnetAttr.confidence = currentMax;
            }
            else
            {
                resnetAttr.attrValue = attributeValues[1];
                resnetAttr.confidence = 1 - currentMax;
            }
        }
        else
        {
            resnetAttr.attrValue = attributeValueStr;
            resnetAttr.confidence = currentMax;
        }
        objectResnetAttr.push_back(resnetAttr);
    }

    attributeResVec.push_back(objectResnetAttr);
    result.clear();

    return APP_ERR_OK;
}

void ResnetAttributePostProcess::MakeAttributeMap(std::ifstream &in, std::string &stringRead)
{
    for (size_t i = 0; i < attributeNum_; ++i)
    {
        MakeNameMap(in, stringRead);
    }
    size_t valueSize = 0;
    for (size_t i = 0; i < attributionIndex_.size(); i++)
    {
        valueSize += attributionIndex_[i].size();
    }
    for (size_t i = 0; i < valueSize; ++i)
    {
        MakeValueMap(in, stringRead);
    }
}

void ResnetAttributePostProcess::MakeNameMap(std::ifstream &in, std::string &stringRead)
{
    if (std::getline(in, stringRead))
    {
        size_t eraseIndex = stringRead.find_last_not_of("\r\n\t");
        if (eraseIndex != std::string::npos)
        {
            stringRead.erase(eraseIndex + 1, stringRead.size() - eraseIndex);
        }
        attributeNameVec_.emplace_back(stringRead);
    }
    else
    {
        attributeNameVec_.emplace_back("");
    }
}

void ResnetAttributePostProcess::MakeValueMap(std::ifstream &in, std::string &stringRead)
{
    if (std::getline(in, stringRead))
    {
        size_t eraseIndex = stringRead.find_last_not_of("\r\n\t");
        if (eraseIndex != std::string::npos)
        {
            stringRead.erase(eraseIndex + 1, stringRead.size() - eraseIndex);
        }
        attributeValueVec_.emplace_back(stringRead);
    }
    else
    {
        attributeValueVec_.emplace_back("");
    }
}

APP_ERROR ResnetAttributePostProcess::Init(std::string &configPath, std::string &labelPath)
{
    configMode_ = MxBase::CONFIGFILE;

    APP_ERROR ret = util_.LoadConfiguration(configPath, configData_, configMode_);
    if (ret != APP_ERR_OK)
    {
        return ret;
    }

    configData_.GetFileValueWarn<size_t>("ATTRIBUTE_NUM", attributeNum_, (size_t)0x0, (size_t)0x3e8);
    configData_.GetFileValueWarn<std::string>("ACTIVATION_FUNCTION", activationFunc_);
    std::string str;
    configData_.GetFileValueWarn<std::string>("ATTRIBUTE_INDEX", str);

    ret = GetAttributeIndex(str);
    if (ret != APP_ERR_OK)
    {
        LogError << GetError(ret) << "Failed to get attribute index.";
        return ret;
    }

    char fullPath[PATH_MAX + 1] = {0x00};
    realpath(labelPath.c_str(), fullPath);

    std::ifstream in;
    in.open(fullPath, std::ios_base::in);
    if (in.fail())
    {
        LogError << GetError(ret) << "Failed to load label file.";
        return APP_ERR_COMM_OPEN_FAIL;
    }

    std::string stringRead;
    std::getline(in, stringRead);

    MakeAttributeMap(in, stringRead);
    in.close();
    LogInfo << "End to initialize ResNetAttributePostProcessor.";

    return APP_ERR_OK;
}

ResnetAttributePostProcess::ResnetAttributePostProcess() {}
