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

#ifndef VIDEOGESTUREREASONER_FRAMESKIPPINGSAMPLING_H
#define VIDEOGESTUREREASONER_FRAMESKIPPINGSAMPLING_H

#include "MxBase/ErrorCode/ErrorCode.h"

namespace AscendFrameSkippingSampling {

    class FrameSkippingSampling {
    public:
        FrameSkippingSampling() = default;

        ~FrameSkippingSampling() = default;

        APP_ERROR Init(uint32_t maxSamplingInterval, uint32_t samplingInterval, uint32_t deviceId);

        APP_ERROR DeInit();

        APP_ERROR Process();

    public:
        bool stopFlag;

    private:
        static uint32_t samplingCounter;
        // device id
        uint32_t deviceId;
        // Sampling interval
        uint32_t samplingInterval;
        // max Sampling interval
        uint32_t maxSamplingInterval;
    };
}// end AscendFrameSkippingSampling
#endif //MULTICHANNELVIDEODETECTION_FRAMESKIPPINGSAMPLING_H
