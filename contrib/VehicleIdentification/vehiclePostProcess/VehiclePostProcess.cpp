/*
 * Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

#include "VehiclePostProcess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
namespace MxBase {
    VehiclePostProcess& VehiclePostProcess::operator=(const VehiclePostProcess &other)
    {
        if (this == &other) {
            return *this;
        }
        ClassPostProcessBase::operator=(other);
        softmax_ = other.softmax_;
        classNum_ = other.classNum_;
        topK_ = other.topK_;
        return *this;
    }

    VehiclePostProcess::VehiclePostProcess(const VehiclePostProcess &other)
    {
        softmax_ = other.softmax_;
        classNum_ = other.classNum_;
        topK_ = other.topK_;
    }

    APP_ERROR VehiclePostProcess::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig)
    {
        LogDebug << "Start to Init VehiclePostProcess.";
        APP_ERROR ret = ClassPostProcessBase::Init(postConfig);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Fail to superInit in ClassPostProcessBase.";
            return ret;
        }
        configData_.GetFileValue("SOFTMAX", softmax_);
        configData_.GetFileValue("CLASS_NUM", classNum_);
        configData_.GetFileValue("TOP_K", topK_);
        LogDebug << "End to Init VehiclePostProcess.";
        return APP_ERR_OK;
    }

    APP_ERROR VehiclePostProcess::DeInit()
    {
        return APP_ERR_OK;
    }

    bool VehiclePostProcess::IsValidTensors(const std::vector<TensorBase> &tensors) const
    {
        const uint32_t softmaxTensorIndex = 0;
        auto softmaxTensor = tensors[softmaxTensorIndex];
        auto softmaxShape = softmaxTensor.GetShape();
        if (softmaxShape[1] != classNum_) {
            LogError << "input size(" << softmaxShape[1] << ") " << "classNumber(" << classNum_ << ")";
            return false;
        }
        return true;
    }

    APP_ERROR VehiclePostProcess::Process(const std::vector<TensorBase>& tensors,
                                          std::vector<std::vector<ClassInfo>> &classInfos,
                                          const std::map<std::string, std::shared_ptr<void>> &configParamMap)
    {
        LogDebug << "Start to Process VehiclePostProcess.";
        APP_ERROR ret = APP_ERR_OK;
        auto inputs = tensors;
        ret = CheckAndMoveTensors(inputs);
        if (ret != APP_ERR_OK) {
            LogError << "CheckAndMoveTensors failed. ret=" << ret;
            return ret;
        }

        if (inputs.size() <= 0) {
            LogError << "The tensors size is invalid!";
            return ret;
        }
        
        const uint32_t softmaxTensorIndex = 0;
        auto softmaxTensor = inputs[softmaxTensorIndex];
        auto shape = softmaxTensor.GetShape();
        uint32_t batchSize = shape[0];
        void *softmaxTensorPtr = softmaxTensor.GetBuffer();
        uint32_t topk = std::min(topK_, classNum_);

        for (uint32_t i = 0; i < batchSize; i++) {
            std::vector<uint32_t> idx = {};
            for (uint32_t j = 0; j < classNum_; j++) {
                idx.push_back(j);
            }

            std::vector<float> softmax = {};
            for (uint32_t j = 0; j < classNum_; j++) {
                float value = *((float*)softmaxTensorPtr + i * classNum_ + j);
                softmax.push_back(value);
            }
            if (softmax_) {
                fastmath::softmax(softmax);
            }

            auto cmp = [&softmax] (uint32_t index_1, uint32_t index_2) {
                return softmax[index_1] > softmax[index_2];
            };

            std::sort(idx.begin(), idx.end(), cmp);

            std::vector<ClassInfo> topkClassInfos = {};
            for (uint32_t j = 0; j < topk; j++) {
                ClassInfo clsInfo = {};
                clsInfo.classId = idx[j];
                clsInfo.confidence = softmax[idx[j]];
                clsInfo.className = configData_.GetClassName(idx[j]);
                topkClassInfos.push_back(clsInfo);
            }
            classInfos.push_back(topkClassInfos);
        }
        LogDebug << "End to Process VehiclePostProcess.";
        return APP_ERR_OK;
    }

    extern "C" {
    std::shared_ptr<MxBase::VehiclePostProcess> GetClassInstance()
    {
        LogInfo << "Begin to get VehiclePostProcess instance.";
        auto instance = std::make_shared<MxBase::VehiclePostProcess>();
        LogInfo << "End to get VehiclePostProcess instance.";
        return instance;
    }
    }
}