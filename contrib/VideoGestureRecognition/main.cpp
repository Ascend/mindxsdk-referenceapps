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

#include "MultiChannelVideoReasoner/MultiChannelVideoReasoner.h"

namespace {
    // device id
    const uint32_t DEVICE_ID = 0;
    // channel id
    const uint32_t BASE_CHANNEL_ID = 0;
}

bool MultiChannelVideoReasoner::forceStop = false;
static void SigHandler(int signal)
{
    if (signal == SIGINT) {
        MultiChannelVideoReasoner::forceStop = true;
        LogInfo << "Force quit MultiChannelVideoReasoner.";
    }
}

int main(int argc, char* argv[])
{
    // rtsp video string
    std::vector<std::string> rtspList = {};

    // load arguments
    std::string rtspPrefix = "rtsp";
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], rtspPrefix.c_str()) == 0 ) {
            auto rtspIndex = strtok(argv[i], reinterpret_cast<const char *>('='));
            auto rtspStream = strtok(NULL, "=");

            LogInfo << rtspIndex << " = " << rtspStream;
            rtspList.emplace_back(rtspStream);
        }
    }
    if (rtspList.empty()) {
//        rtspList.emplace_back("./test1.mp4");
//        rtspList.emplace_back("rtsp://192.168.88.109:31951/test4.264");
//        rtspList.emplace_back("rtsp://192.168.88.109:31854/gesture_test.264");
        rtspList.emplace_back("./test.mp4");
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

    auto multiChannelVideoReasoner = std::make_shared<MultiChannelVideoReasoner>();
    ReasonerConfig reasonerConfig;
    reasonerConfig.deviceId = DEVICE_ID;
    reasonerConfig.baseVideoChannelId = BASE_CHANNEL_ID;
    reasonerConfig.rtspList = rtspList;
    reasonerConfig.maxTryOpenVideoStream = 10;
    reasonerConfig.resnetModelPath = "./model/resnet18.om";
    reasonerConfig.resnetLabelPath = "./model/resnet18.names";
    reasonerConfig.resnetModelWidth = 256;//416;
    reasonerConfig.resnetModelHeight = 224;//416;
    reasonerConfig.maxDecodeFrameQueueLength = 100;
    reasonerConfig.popDecodeFrameWaitTime = 10;
    reasonerConfig.SamplingInterval = 24;
    reasonerConfig.maxSamplingInterval = 100;

    // init
    ret = multiChannelVideoReasoner->Init(reasonerConfig);
    if (ret != APP_ERR_OK) {
        LogError << "Init multi channel video infer failed.";
        return ret;
    }

    // run
    multiChannelVideoReasoner->Process();

    // destroy reasoner
    ret = multiChannelVideoReasoner->DeInit();
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

    return 0;
}








