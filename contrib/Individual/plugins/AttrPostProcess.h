/*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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
#include "MxBase/PostProcessBases/ClassPostProcessBase.h"

namespace MxBase {
    class AttrPostProcess : public ClassPostProcessBase {
    public:
        AttrPostProcess() = default;

        ~AttrPostProcess() = default;

        AttrPostProcess(const AttrPostProcess &other) = default;

        APP_ERROR Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) override;

        APP_ERROR DeInit() override;

        APP_ERROR Process(const std::vector<TensorBase> &tensors, std::vector<std::vector<ClassInfo>> &classInfos,
                          const std::map<std::string, std::shared_ptr<void>> &configParamMap = {}) override;

        AttrPostProcess &operator=(const AttrPostProcess &other);

        bool IsValidTensors(const std::vector<TensorBase> &tensors) const override;

    private:
        uint32_t classNum_ = 0;
        bool softMax = true;
        uint32_t topK_ = 40;
    };

    extern "C" {
    std::shared_ptr<MxBase::AttrPostProcess> GetClassInstance();
    }
}
#endif
