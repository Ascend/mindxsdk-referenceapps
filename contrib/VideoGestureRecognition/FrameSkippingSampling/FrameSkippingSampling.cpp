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

#include "FrameSkippingSampling.h"
#include "MxBase/Log/Log.h"

namespace AscendFrameSkippingSampling {
    uint32_t FrameSkippingSampling::samplingCounter = 0;

    APP_ERROR FrameSkippingSampling::Init(uint32_t maxSamplingInterval,
                                          uint32_t samplingInterval,
                                          uint32_t deviceId)
    {
        LogDebug << "FrameSkippingSampling" << ": FrameSkippingSampling init start.";

        stopFlag = false;
        this->maxSamplingInterval = maxSamplingInterval;
        this->samplingInterval = samplingInterval;
        this->deviceId = deviceId;

        LogDebug << "FrameSkippingSampling" << ": FrameSkippingSampling init success.";
        return APP_ERR_OK;
    }

    APP_ERROR FrameSkippingSampling::DeInit()
    {
        LogDebug << "FrameSkippingSampling" << ": FrameSkippingSampling deinit start.";

        stopFlag = true;
        samplingInterval = 1;

        LogDebug << "FrameSkippingSampling" << ": FrameSkippingSampling deinit success.";
        return APP_ERR_OK;
    }

    APP_ERROR FrameSkippingSampling::Process()
    {
        stopFlag = false;
        if (samplingInterval > maxSamplingInterval) {
            LogError << "sample interval exceeding the upper limit";
            return APP_ERR_COMM_FAILURE;
        }
        if (samplingCounter % samplingInterval == 0) {
            stopFlag = true;
        }
        samplingCounter += 1;
        return APP_ERR_OK;
    }
}
