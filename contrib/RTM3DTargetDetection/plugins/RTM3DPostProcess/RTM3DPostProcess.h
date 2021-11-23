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

#ifndef RTM3D_POST_PROCESS_H
#define RTM3D_POST_PROCESS_H
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

// 在不同的名字空间下可以存在同名的函数、类、变量，它们互不影响    using namespace std方便调用cout、endl等所以不容易发现其实这个习惯可能会出现问题

namespace MxBase {      // MxBase是命名空间名字
    class RTM3DPostProcess : public ObjectPostProcessBase
    {
    public:
        RTM3DPostProcess() = default;   // 无参

        ~RTM3DPostProcess() = default;  // 析构函数 销毁

        RTM3DPostProcess(const RTM3DPostProcess &other) = default;  // 带参

        RTM3DPostProcess &operator=(const RTM3DPostProcess &other) = default;    // 重载运算符=  类的对象能互相赋值

        APP_ERROR Init(const std::map <std::string, std::shared_ptr<void>> &postConfig) override;

        APP_ERROR DeInit() override;

        APP_ERROR Process(const std::vector <MxBase::TensorBase> &tensors,
                          std::vector <std::vector<MxBase::ObjectInfo>> &objectInfos,
                          const std::vector <MxBase::ResizedImageInfo> &resizedImageInfos = {},
                          const std::map <std::string, std::shared_ptr<void>> &paramMap = {}) override;

    protected:
        bool IsValidTensors(const std::vector <MxBase::TensorBase> &tensors) const override;

        void ObjectDetectionOutput(const std::vector <MxBase::TensorBase> &tensors,
                                   std::vector <std::vector<MxBase::ObjectInfo>> &objectInfos);

        void GenerateBbox(std::vector <MxBase::ObjectInfo> &detBoxes,
                          std::vector <std::shared_ptr<void>> featLayerData,
                          const std::vector <std::vector<size_t>> &featLayerShapes);
    protected:
        int RTM3DType_ = 4;
        bool RTM3DUsing_ = true;

    };
    extern "C" {
    std::shared_ptr<MxBase::RTM3DPostProcess> GetObjectInstance();
    }
}
#endif