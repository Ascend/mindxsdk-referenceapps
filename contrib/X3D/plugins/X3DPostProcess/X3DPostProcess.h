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
#ifndef X3D_POST_PROCESS
#define X3D_POST_PROCESS
#include "MxBase/PostProcessBases/ClassPostProcessBase.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"

namespace MxBase {
class X3DPostProcess : public ClassPostProcessBase {
public:
    X3DPostProcess() = default;

    ~X3DPostProcess() = default;

    X3DPostProcess(const X3DPostProcess &other);

    APP_ERROR Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) override;

    APP_ERROR DeInit() override;

    APP_ERROR Process(const std::vector<TensorBase> &tensors, std::vector<std::vector<ClassInfo>> &classInfos,
                      const std::map<std::string, std::shared_ptr<void>> &configParamMap = {}) override;

    X3DPostProcess &operator=(const X3DPostProcess &other);

    bool IsValidTensors(const std::vector<TensorBase> &tensors) const;

private:
    uint32_t classNum_ = 0;
    bool softmax_ = true;
    uint32_t topK_ = 1;
};

extern "C" {
std::shared_ptr<MxBase::X3DPostProcess> GetClassInstance();
}
}
#endif