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

#include <iostream>
#include <vector>
#include <algorithm>
#include "ResnetFeaturePostProcess.h"

float ResnetFeaturePostProcess::ActivateOutput(float data, bool isAct)
{
    if (isAct == true)
    {
        return fastmath::sigmoid(data);
    }
    else
    {
        return data;
    }
}

APP_ERROR ResnetFeaturePostProcess::ResnetfeaturePostProcess(std::vector<MxBase::Tensor>& inferOutputs, std::vector<float>& features, bool isSigmoid)
{
   
    if (inferOutputs.empty())
    {
        LogError << "result Infer failed with empty output..." << std::endl;
        return APP_ERR_INVALID_PARAM;
    }

    size_t featureSize = inferOutputs[0].size / FEATURE_SIZE;
    float *castData = static_cast<float *>(inferOutputs[0].GetData());
    
    for (size_t i = 0; i < featureSize; i++)
    {
        features.push_back(ActivateOutput(castData[i], isSigmoid));
    }
    return APP_ERR_OK;
}