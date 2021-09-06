/**
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
 **/

#ifndef RCF_POST_PROCESS_H
#define RCF_POST_PROCESS_H
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "opencv2/opencv.hpp"


class RcfPostProcess : public MxBase::ObjectPostProcessBase {
public:
    RcfPostProcess() = default;
    ~RcfPostProcess() = default;
    RcfPostProcess &operator=(const RcfPostProcess &other) = default;
    APP_ERROR Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) override;
    APP_ERROR DeInit() override;
    APP_ERROR Process(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);

protected:
    int outSizeNum = 5;
    int rcfType = 5;
    int modelType = 0;
    int inputType = 0;
};
#endif
