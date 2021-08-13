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

#include "MxBase/Log/Log.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/ConfigUtil/ConfigUtil.h"

#include "MultiChannelVideoReasoner/MultiChannelVideoReasoner.h"

namespace {
    // device id
    const uint32_t DEVICE_ID = 0;
    // channel id
    const uint32_t BASE_CHANNEL_ID = 0;
}

bool MultiChannelVideoReasoner::_s_force_stop = false;
static void SigHandler(int signal)
{
    if (signal == SIGINT) {
        MultiChannelVideoReasoner::_s_force_stop = true;
        LogInfo << "Force quit MultiChannelVideoReasoner.";
    }
}

int main(int argc, char* argv[])
{
    // rtsp video string
    std::vector<std::string> rtspList = {};
    rtspList.emplace_back("#{rtsp流地址1}");
    rtspList.emplace_back("#{rtsp流地址2}");

    ///=== modify config ===//
    MxBase::ConfigData configData;
    MxBase::ConfigUtil configUtil;
    APP_ERROR ret = configUtil.LoadConfiguration("${MindXSDK安装路径}/config/logging.conf",
                                 configData, MxBase::ConfigMode::CONFIGFILE);
    if (ret == APP_ERR_OK) {
        configData.SetFileValue<int>("global_level", 1);
        MxBase::Log::SetLogParameters(configData);
    } else {
        LogInfo << "load log configuration failed.";
    }

    ///=== resource init ===///
    // init devices
    ret = MxBase::DeviceManager::GetInstance()->InitDevices();
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
    reasonerConfig.yoloModelPath = "${yolov3.om模型路径}";
    reasonerConfig.yoloLabelPath = "${yolov3 coco.names路径}";
    reasonerConfig.yoloModelWidth = 416;
    reasonerConfig.yoloModelHeight = 416;
    reasonerConfig.maxDecodeFrameQueueLength = 400;
    reasonerConfig.popDecodeFrameWaitTime = 10;
    reasonerConfig.intervalPerformanceMonitorPrint = 5;
    reasonerConfig.intervalMainThreadControlCheck = 2;
    reasonerConfig.writeDetectResultToFile = false;
    reasonerConfig.enablePerformanceMonitorPrint = true;

    // init
    ret = multiChannelVideoReasoner->Init(reasonerConfig);
    if (ret != APP_ERR_OK) {
        LogError << "Init multi channel video infer failed.";
        MxBase::DeviceManager::GetInstance()->DestroyDevices();
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








