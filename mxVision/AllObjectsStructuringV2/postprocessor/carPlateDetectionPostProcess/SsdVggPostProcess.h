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

#ifndef SSDVGG_SSDVGGPOSTPROCESS_H
#define SSDVGG_SSDVGGPOSTPROCESS_H
#include "MxBase/MxBase.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

class SsdVggPostProcess {
public:
    SsdVggPostProcess();

    ~SsdVggPostProcess() {}

    APP_ERROR Init(const std::map<std::string, std::string> &postConfig);

    APP_ERROR DeInit();

    APP_ERROR Process(const MxBase::Image& originImage, const std::vector<MxBase::Tensor>& inferOutputs,
                      std::vector<MxBase::ObjectInfo>& objectInfos);

private:
    uint32_t classNum_;
    float scoreThresh_;
    MxBase::ConfigUtil util_;
    MxBase::ConfigData configData_;
};

#endif //SSDVGG_SSDVGGPOSTPROCESS_H

