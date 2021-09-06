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

#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/Log/Log.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "VideoGestureReasoner/VideoGestureReasoner.h"

namespace {
    // device id
    const uint32_t DEVICE_ID = 0;
    // channel id
    const uint32_t BASE_CHANNEL_ID = 0;
    // model input width
    const uint32_t MODEL_WIDTH = 256;
    // model input height
    const uint32_t MODEL_HEIGHT = 224;
    // sampling interval
    const uint32_t SAMPLING_INTERVAL = 24;
    // maximum sampling interval
    const uint32_t MAX_SAMPLING_INTERVAL = 100;
    // decoding waiting time
    const uint32_t DECODE_FRAME_WAIT_TIME = 10;
    // maximum decoding queue length
    const uint32_t DECODE_FRAME_QUEUE_LENGTH = 100;
}

static void SigHandler(int signal)
{
    if (signal == SIGINT) {
        VideoGestureReasoner::forceStop = true;
        LogInfo << "Force quit VideoGestureReasoner.";
    }
}
static APP_ERROR process(std::vector<std::string> rtspList)
{
    auto videoGestureReasoner = std::make_shared<VideoGestureReasoner>();
    ReasonerConfig reasonerConfig;
    reasonerConfig.deviceId = DEVICE_ID;
    reasonerConfig.baseVideoChannelId = BASE_CHANNEL_ID;
    reasonerConfig.rtspList = rtspList;
    reasonerConfig.resnetModelPath = "${gesture_yuv.om模型路径}";
    reasonerConfig.resnetLabelPath = "${resnet18.names路径}";
    reasonerConfig.resnetModelWidth = MODEL_WIDTH;
    reasonerConfig.resnetModelHeight = MODEL_HEIGHT;
    reasonerConfig.maxDecodeFrameQueueLength = DECODE_FRAME_QUEUE_LENGTH;
    reasonerConfig.popDecodeFrameWaitTime = DECODE_FRAME_WAIT_TIME;
    reasonerConfig.samplingInterval = SAMPLING_INTERVAL;
    reasonerConfig.maxSamplingInterval = MAX_SAMPLING_INTERVAL;

    // init
    APP_ERROR ret = videoGestureReasoner->Init(reasonerConfig);
    if (ret != APP_ERR_OK) {
        LogError << "Init multi channel video infer failed.";
        return ret;
    }

    // run
    videoGestureReasoner->Process();

    // destroy reasoner
    ret = videoGestureReasoner->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "Deinit multi channel video infer failed.";
        return ret;
    }

    // destroy devices
    ret = MxBase::DeviceManager::GetInstance()->DestroyDevices();
    if (ret != APP_ERR_OK) {
        LogError << "DestroyDevices failed";
        return ret;
    }
    return APP_ERR_OK;
}

int main(int argc, char *argv[])
{
    // rtsp video string
    std::vector<std::string> rtspList = {};

    // load arguments
    std::string rtspPrefix = "rtsp";
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], rtspPrefix.c_str()) == 0) {
            auto rtspIndex = strtok(argv[i], reinterpret_cast<const char *>('='));
            auto rtspStream = strtok(NULL, "=");

            LogInfo << rtspIndex << " = " << rtspStream;
            rtspList.emplace_back(rtspStream);
        }
    }
    if (rtspList.empty()) {
        rtspList.emplace_back("#{rtsp流地址}");
    }

    ///=== resource init ===///
    // init devices
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "InitDevices failed";
        return ret;
    }

    // set devices
    MxBase::DeviceContext device;
    device.devId = DEVICE_ID;
    ret = MxBase::DeviceManager::GetInstance()->SetDevice(device);
    if (ret != APP_ERR_OK) {
        LogError << "SetDevice failed";
        return ret;
    }

    if (signal(SIGINT, SigHandler) == SIG_ERR) {
        LogError << "can not catch SIGINT";
        return APP_ERR_COMM_FAILURE;
    }

    // inference start
    ret = process(rtspList);
    if (ret != APP_ERR_OK) {
        LogError << "inference start failed";
        return ret;
    }
    return 0;
}