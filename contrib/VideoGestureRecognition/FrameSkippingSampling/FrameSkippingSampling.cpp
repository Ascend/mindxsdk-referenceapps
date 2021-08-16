//
// Created by 615 on 2021/8/10.
//

#include "FrameSkippingSampling.h"

#include "MxBase/Log/Log.h"

namespace AscendFrameSkippingSampling {
    uint32_t FrameSkippingSampling::SamplingCounter = 0;

    APP_ERROR FrameSkippingSampling::Init(uint32_t maxSamplingInterval, uint32_t SamplingInterval,
                                                             uint32_t deviceId) {
        LogDebug << "FrameSkippingSampling" << ": FrameSkippingSampling init start.";

        stopFlag = false;
        this->maxSamplingInterval = maxSamplingInterval;
        this->SamplingInterval = SamplingInterval;
        this->deviceId = deviceId;

        LogDebug << "FrameSkippingSampling" << ": FrameSkippingSampling init success.";
        return APP_ERR_OK;
    }

    APP_ERROR FrameSkippingSampling::DeInit() {
        LogDebug << "FrameSkippingSampling" << ": FrameSkippingSampling deinit start.";

        stopFlag = true;
        SamplingInterval = 1;

        LogDebug << "FrameSkippingSampling" << ": FrameSkippingSampling deinit success.";
        return APP_ERR_OK;
    }

    APP_ERROR FrameSkippingSampling::Process() {
        stopFlag = false;
        if (SamplingInterval > maxSamplingInterval) {
            LogError << "sample interval exceeding the upper limit";
            return APP_ERR_COMM_FAILURE;
        }
        if (SamplingCounter % SamplingInterval == 0) {
            stopFlag = true;
        }
        SamplingCounter += 1;
        return APP_ERR_OK;
    }
}
