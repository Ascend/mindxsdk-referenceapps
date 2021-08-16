//
// Created by 615 on 2021/8/10.
//

#ifndef MULTICHANNELVIDEODETECTION_FRAMESKIPPINGSAMPLING_H
#define MULTICHANNELVIDEODETECTION_FRAMESKIPPINGSAMPLING_H
#include "MxBase/ErrorCode/ErrorCode.h"
namespace AscendFrameSkippingSampling {

    class FrameSkippingSampling {
    public:
        FrameSkippingSampling() = default;

        ~FrameSkippingSampling() = default;

        APP_ERROR Init(uint32_t maxSamplingInterval, uint32_t SamplingInterval, uint32_t deviceId);

        APP_ERROR DeInit();

        APP_ERROR Process();

    public:
        bool stopFlag;

    private:
        static uint32_t SamplingCounter;
        // device id
        uint32_t deviceId;
        // Sampling interval
        uint32_t SamplingInterval;
        // max Sampling interval
        uint32_t maxSamplingInterval;
    };
}// end AscendFrameSkippingSampling
#endif //MULTICHANNELVIDEODETECTION_FRAMESKIPPINGSAMPLING_H
