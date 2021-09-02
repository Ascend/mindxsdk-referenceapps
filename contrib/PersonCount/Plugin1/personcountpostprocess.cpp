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
#include<iostream>
#include<fstream>
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
#include "personcountpostprocess.h"
namespace MxBase {
APP_ERROR CountPersonPostProcessor::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig)
{
    LogDebug << "Start to Init SamplePostProcess.";
    APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
        return ret;
    }
    LogDebug << "End to Init SamplePostProcess.";
    return APP_ERR_OK;
}

APP_ERROR CountPersonPostProcessor::DeInit()
{
    return APP_ERR_OK;
}

bool CountPersonPostProcessor::IsValidTensors(const std::vector<TensorBase> &tensors) const
{
    return true;
}
APP_ERROR CountPersonPostProcessor::Process(const std::vector<TensorBase>& tensors, 
                                            std::vector< std::vector<ObjectInfo>>& objectInfos,
                                            const std::vector<ResizedImageInfo>& resizedImageInfos,
                                            const std::map<std::string, std::shared_ptr<void>> &configParamMap)
{
    LogDebug << "Start to Process CountPersonPostProcessor.";
    APP_ERROR ret = APP_ERR_OK;
    auto inputs = tensors;
    ret = CheckAndMoveTensors(inputs);
    if (ret != APP_ERR_OK) {
        LogError << "CheckAndMoveTensors failed. ret=" << ret;
        return ret;
    }
    const uint32_t graphTensorIndex = 0;
    auto graphTensor = inputs[graphTensorIndex];
    auto shape = graphTensor.GetShape();
    uint32_t batchSize = shape[0];
    float *graphTensorPtr = (float*)(graphTensor.GetBuffer());
    std::vector<uint8_t> temp;
    ObjectInfo tempobject;
    std::vector<ObjectInfo> mid_objectinfo;
    for (uint32_t b = 0; b < batchSize; b++) {
        graphTensorPtr += b*image_H*image_W;
        float sum = 0;
        float max = *((float*)graphTensorPtr);
        float min = *((float*)graphTensorPtr);
        uint32_t i = 0;
        for (i = 0; i < image_H * image_W; i++) {
            float value = *(graphTensorPtr + i);
            sum += value;
            if (value > max) {
                max = value;
            }
            if (value < min) {
                min = value;
            }
        }
        for (i = 0; i < image_H * image_W; i++) {
            float value = *(graphTensorPtr + i);
            uint8_t ivalue = int(image_scale_factor * (value - min) / (max - min));
            temp.push_back(ivalue);
        }
        tempobject.mask.push_back(temp);
        tempobject.classId = sum / person_numper_scale_factor;
        mid_objectinfo.push_back(tempobject);
    }
    objectInfos.push_back(mid_objectinfo);
    LogDebug << "End to Process SamplePostProcess.";
    return APP_ERR_OK;
}

extern "C" {
std::shared_ptr<MxBase::CountPersonPostProcessor> GetObjectInstance()
{
    LogInfo << "Begin to get CountPersonPostProcessor instance.";
    auto instance = std::make_shared<MxBase::CountPersonPostProcessor>();
    LogInfo << "End to get SamplePostProcess instance.";
    return instance;
}
}
}