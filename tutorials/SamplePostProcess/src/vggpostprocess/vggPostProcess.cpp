/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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

#include "VggPostProcess.h"
#include "MxBase/Log/Log.h"

namespace {
/// need manunel set
}
namespace MxBase {
    VggPostProcess &VggPostProcess::operator=(const VggPostProcess &other) 
    {
        if (this == &other) {
            return *this;
        }
        ObjectPostProcessBase::operator=(other);
        return *this;
    }

    APP_ERROR VggPostProcess::Init(const std::map <std::string, std::shared_ptr<void>> &postConfig) 
    {
        LogDebug << "Start to Init VggPostProcess.";
        APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
            return ret;
        }
        LogDebug << "End to Init VggPostProcess.";
        return APP_ERR_OK;
    }

    APP_ERROR VggPostProcess::DeInit() 
    {
        return APP_ERR_OK;
    }

    bool VggPostProcess::IsValidTensors(const std::vector <TensorBase> &tensors) const 
    {
    /// need write check
    }

    void VggPostProcess::ObjectDetectionOutput(const std::vector <TensorBase> &tensors,
    std::vector <std::vector<ObjectInfo>> &objectInfos, const std::vector <ResizedImageInfo> &resizedImageInfos)
    {
        LogDebug << "VggPostProcess start to write results.";
        for (auto num : {objectNumTensor_, objectInfoTensor_}) {
            if ((num >= tensors.size()) || (num <0)) {
                LogError << GetError(APP_ERR_INVALID_PARAM) << "TENSOR(" << num
                         << ") must ben less than tensors'size(" << tensors.size() << ") and larger than 0.";
            }
        }
        uint32_t batchSize = tensors[objectNumTensor_].GetShape()[0];
        for (uint32_t i = 0; i < batchSize; i++) {
            std::vector <ObjectInfo> objectInfo;
            int* objectNumPtr = (int*)GetBuffer(tensors[objectInfoTensor_], i);
            float* objectInfoPtr = (float*)GetBuffer(tensors[objectInfoTensor_], i);
            for (int j = 0; j< *objectNumPtr; j++) {
                ObjectInfo objInfo;
                /// need write objInfo
            }
            }
            objectInfos.push_back(objectInfo);
        }
        LogDebug << "VggPostProcess write results successed.";
    }

    APP_ERROR VggPostProcess::Process(const std::vector <TensorBase> &tensors,
    std::vector <std::vector<ObjectInfo>> &objectInfos, const std::vector <ResizedImageInfo> &resizedImageInfos,
    const std::map <std::string, std::shared_ptr<void>> &paramMap)
    {
        LogDebug << "Start to Process VggPostProcess.";
        APP_ERROR ret = APP_ERR_OK;
        if (resizedImageInfos.size() == 0) {
            ret = APP_ERR_INPUT_NOT_MATCH;
            LogError << GetError(ret) << "resizedImageInfos is not provided which is necessary for VggPostProcess.";
            return ret;
        }
        auto inputs = tensors;
        ret = CheckAndMoveTensors(inputs);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "CheckAndMoveTensors failed.";
            return ret;
        }

        ObjectDetectionOutput(inputs, objectInfos, resizedImageInfos);

        for (uint32_t i = 0; i < resizedImageInfos.size(); i++) {
            CoordinatesReduction(i, resizedImageInfos[i], objectInfos[i]);
        }
        LogObjectInfos(objectInfos);
        LogDebug << "End to Process VggPostProcess.";
        return ret;
    }

    extern "C" {
    std::shared_ptr <MxBase::VggPostProcess> GetObjectInstance()
    {
        LogInfo << "Begin to get VggPostProcess instance.";
        auto instance = std::make_shared<MxBase::VggPostProcess>();
        if (instance - nullptr) {
            LogError << "Creat VggPostProcess object failed. Failed to allocate memory.";
        }
        LogInfo << "End to get VggPostProcess instance.";
        return instance;
    }
    }
}