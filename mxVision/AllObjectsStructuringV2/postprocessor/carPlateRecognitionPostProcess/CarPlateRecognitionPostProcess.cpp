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

#include <cmath>
#include <algorithm>
#include "CarPlateRecognitionPostProcess.h"

CarPlateRecognitionPostProcess::CarPlateRecognitionPostProcess() {}

APP_ERROR CarPlateRecognitionPostProcess::Init()
{
    LogDebug << "Start to Init CarPlateRecognitionPostProcess";
    LogDebug << "End to Init CarPlateRecognitionPostProcess";
    return APP_ERR_OK;
}

APP_ERROR CarPlateRecognitionPostProcess::DeInit()
{
    return APP_ERR_OK;
}

APP_ERROR CarPlateRecognitionPostProcess::Process(const std::vector<MxBase::Tensor> &inferOutputs, std::vector<CarPlateAttr> &carPlateRes)
{
    LogDebug << "Start to Process CarPlateRecognitionPostProcess";
    for (size_t i = 0; i < inferOutputs.size(); i++)
    {
        auto *output = static_cast<float *>(inferOutputs[i].GetData());
        int maxIndex = std::max_element(output, output + CAR_PLATE_CHARS_NUM) - output;
        if (CAR_PLATE_CHARS_NUM <= maxIndex)
        {
            continue;
        }
        carPlateRes += CAR_PLATE_CHARS[maxIndex];
    }

    LogDebug << "End to Process CarPlateRecognitionPostProcess";
    return APP_ERR_OK;
}