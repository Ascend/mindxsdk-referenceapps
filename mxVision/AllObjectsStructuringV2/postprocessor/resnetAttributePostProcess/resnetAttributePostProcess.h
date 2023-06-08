/*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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

#ifndef MAIN_RESNETATTRIBUTEPOSTPROCESS_H
#define MAIN_RESNETATTRIBUTEPOSTPROCESS_H

#include "MxBase/MxBase.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/ConfigUtil/ConfigUtil.h"
#include "MxBase/Maths/FastMath.h"

struct ResnetAttr
{
    uint32_t attId;
    std::string attrName;
    std::string attrValue;
    float confidence;
};

float CONFIDENCE = 0.5;

class ResnetAttributePostProcess
{
private:
    site_t attrbuteNum_;
    std::string activationFunc_;
    std::vector<std::vector<int>> attributionIndex_;
    MxBase::ConfigData configData_;
    MxBase::ConfigUtil util_;
    MxBase::ConfigMode configMode_;
    std::vector < std::string >> attributeNameVec_;
    std::vector < std::string >> attributeValueVec_;

    APP_ERROR GetAttributeIndex(std::string &attrAttributeIndex);

    void MakeAttributeMap(std::ifstream &in, std::string &stringRead);

    void MakeNameMap(std::ifstream &in, std::string &stringRead);

    void MakeValueMap(std::ifstream &in, std::string &stringRead);

public:
    resnetAttributePostProcess();

    ~resnetAttributePostProcess() = default;

    APP_ERROR Init(std::string configPath, std::string labelPath);

    App_ERROR Process(std::vector<MxBase::Tensor> &resnetInferResVec, std::vecor<std::vector<ResnetAttr>> &attributeResVec);
};

#endif