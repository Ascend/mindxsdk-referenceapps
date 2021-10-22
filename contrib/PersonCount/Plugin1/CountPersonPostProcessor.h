/*
 * Copyright (c) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SAMPLE_POST_PROCESS_H
#define SAMPLE_POST_PROCESS_H
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

namespace MxBase {
class CountPersonPostProcessor : public ObjectPostProcessBase {
public:
    CountPersonPostProcessor() = default;

    ~CountPersonPostProcessor() = default;

    CountPersonPostProcessor(const CountPersonPostProcessor & other) = default;

    APP_ERROR Init(const std::map<std::string, std::shared_ptr<void>> & postConfig) override;

    APP_ERROR DeInit() override;
    bool IsValidTensors(const std::vector<TensorBase> & tensors) const;
    APP_ERROR Process(const std::vector<TensorBase> & tensors, std::vector<std::vector<ObjectInfo>> & objectInfos,
                              const std::vector<ResizedImageInfo> & resizedImageInfos = {},
                              const std::map<std::string, std::shared_ptr<void>> & configParamMap = {}) override;

private:
    uint32_t imageH = 800; /* the height  of outpu feature map */
    uint32_t imageW = 1408; /* the width  of outpu feature map */
    uint32_t imageScaleFactor = 255;
    uint32_t personNumberScaleFactor = 1000;
};

extern "C" {
std::shared_ptr<MxBase::CountPersonPostProcessor> GetObjectInstance();
}
}
#endif