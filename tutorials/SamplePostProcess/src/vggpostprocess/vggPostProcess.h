/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
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

#ifndef VGG_POST_PROCESS_H
#define VGG_POST_PROCESS_H
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

namespace {
const int DEFAULT_OBJECT_NUM_TENSOR = 0;
const int DEFAULT_OBJECT_INFO_TENSOR = 1;
}

namespace MxBase {
    class VggPostProcess{

    public:
        VggPostProcess() = default;

        ~VggPostProcess() = default;

        VggPostProcess(const VggPostProcess &other) = default;

        VggPostProcess &operator=(const VggPostProcess &other);

        APP_ERROR Init(const std::map <std::string, std::shared_ptr<void>> &postConfig) override;

        APP_ERROR DeInit() override;

        APP_ERROR Process(const std::vector <MxBase::TensorBase> &tensors,
                          std::vector <std::vector<MxBase::ObjectInfo>> &objectInfos,
                          const std::vector <MxBase::ResizedImageInfo> &resizedImageInfos = {},
                          const std::map <std::string, std::shared_ptr<void>> &paramMap = {}) override;

    protected:
        bool IsValidTensors(const std::vector <MxBase::TensorBase> &tensors) const override;

        void ObjectDetectionOutput(const std::vector <MxBase::TensorBase> &tensors,
                                   std::vector <std::vector<MxBase::ObjectInfo>> &objectInfos,
                                   const std::vector <MxBase::ResizedImageInfo> &resizedImageInfos = {});
    private:
        uint32_t objectNumTensor_ = DEFAULT_OBJECT_NUM_TENSOR;
        uint32_t objectInfoTensor_ = DEFAULT_OBJECT_INFO_TENSOR;

    };
    extern "C" {
    std::shared_ptr<MxBase::VggPostProcess> GetObjectInstance();
    }
}
#endif