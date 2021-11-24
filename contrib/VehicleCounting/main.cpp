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
#include <iostream>
#include <memory>
#include <queue>
#include <mutex>
#include <thread>
#include <csignal>
#include <unistd.h>
#include "MxBase/ErrorCode/ErrorCodes.h"
#include "MxBase/Log/Log.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/ConfigUtil/ConfigUtil.h"
#include "BlockingQueue/BlockingQueue.h"
#include "VideoProcess/VideoProcess.h"
#include "Yolov4Detection/Yolov4Detection.h"
#include "ReadConfig/GetConfig.h"

bool VideoProcess::stopFlag = false;
std::vector<double> g_inferCost;
namespace {
    const uint32_t MAX_QUEUE_LENGHT = 2000;
    const uint32_t VIDEO_WIDTH = 1280;
    const uint32_t VIDEO_HEIGHT = 720;
    const uint32_t frame_rate = 15;
}

void SigHandler(int signal)
{
    if (signal == SIGINT) {
        VideoProcess::stopFlag = true;
    }
}

void InitYolov4Param(InitParam &initParam, const uint32_t deviceID)
{
    initParam.deviceId = deviceID;
    initParam.labelPath = "./model/coco.names";
    initParam.checkTensor = true;
    initParam.modelPath = "./model/yolov4_bs.om";
    initParam.classNum = 80;
    initParam.biasesNum = 18;
    initParam.biases = "10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326";
    initParam.objectnessThresh = "0.001";
    initParam.iouThresh = "0.5";
    initParam.scoreThresh = "0.001";
    initParam.yoloType = 3;
    initParam.modelType = 1;
    initParam.inputType = 0;
    initParam.anchorDim = 3;
}

int main() {
    ///=== modify config ===//
    MxBase::ConfigData configData;
    MxBase::ConfigUtil configUtil;
    configUtil.LoadConfiguration("$ENV{MX_SDK_HOME}/config/logging.conf", configData, MxBase::ConfigMode::CONFIGFILE);
    configData.SetFileValue<int>("global_level", 1);
    MxBase::Log::SetLogParameters(configData);
    std::string streamName = "./data/test1.264";
    // read config file
    std::string m_sPath="./params.config";
    std::map<string,string> m_mapConfig;
    bool isfile=ReadConfig(m_sPath,m_mapConfig);
    if(!isfile){
        LogError << "Read config file failed";
        return 0;
    }
    setParams(m_mapConfig);
    setThreshold(m_mapConfig);
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "InitDevices failed";
        return ret;
    }
    auto videoProcess = std::make_shared<VideoProcess>();
    auto yolov4 = std::make_shared<Yolov4Detection>();
    auto tracker = std::make_shared<ascendVehicleTracking::MOTConnection>();
    InitParam initParam;
    InitYolov4Param(initParam, videoProcess->DEVICE_ID);
    // 初始化模型推理所需的配置信息
    ret = yolov4->FrameInit(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "FrameInit failed";
        MxBase::DeviceManager::GetInstance()->DestroyDevices();
        return ret;
    }
    MxBase::DeviceContext device;
    device.devId = videoProcess->DEVICE_ID;
    ret = MxBase::DeviceManager::GetInstance()->SetDevice(device);
    if (ret != APP_ERR_OK) {
        LogError << "SetDevice failed";
        yolov4->FrameDeInit();
        MxBase::DeviceManager::GetInstance()->DestroyDevices();
        return ret;
    }
    // 视频流处理
    ret = videoProcess->StreamInit(streamName);
    if (ret != APP_ERR_OK) {
        LogError << "StreamInit failed";
        yolov4->FrameDeInit();
        MxBase::DeviceManager::GetInstance()->DestroyDevices();
        return ret;
    }

    // 解码模块功能初始化
    ret = videoProcess->VideoDecodeInit();
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecodeInit failed";
        yolov4->FrameDeInit();
        videoProcess->StreamDeInit();
        MxBase::DeviceManager::GetInstance()->DestroyDevices();
        return ret;
    }

    auto blockingQueue = std::make_shared<BlockingQueue<std::shared_ptr<void>>>(MAX_QUEUE_LENGHT);
    std::thread getFrame(videoProcess->GetFrames, blockingQueue, videoProcess);
    std::thread getResult(videoProcess->GetResults, blockingQueue, yolov4, videoProcess, tracker);
    if (signal(SIGINT, SigHandler)) {
        LogError << "can not catch SIGINT";
        return APP_ERR_COMM_FAILURE;
    }

    while (!videoProcess->stopFlag) {
        sleep(10);
    }
    // 生成视频
    LogInfo << "Creating video...";
    std::queue<cv::Mat> video_frames=videoProcess->Getframes();
    cv::VideoWriter writer("./result1/test01.avi", cv::VideoWriter::fourcc('M','J','P','G'), 
                           frame_rate, cv::Size(VIDEO_WIDTH,VIDEO_HEIGHT), true);
    uint32_t frame_count = video_frames.size();
    for(uint32_t i = 0; i<frame_count; i++){
        writer.write(video_frames.front());
        video_frames.pop();
    }
    writer.release();
    LogInfo << "Video creating finish...";
    
    getFrame.join();
    getResult.join();

    blockingQueue->Stop();
    blockingQueue->Clear();
    ret = yolov4->FrameDeInit();
    if (ret != APP_ERR_OK) {
        LogError << "FrameInit failed";
        return ret;
    }
    ret = videoProcess->StreamDeInit();
    if (ret != APP_ERR_OK) {
        LogError << "StreamDeInit failed";
        return ret;
    }
    LogInfo << "ending1...";
    ret = videoProcess->VideoDecodeDeInit();
    LogInfo << "ending2...";
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecodeDeInit failed";
        return ret;
    }
    ret = MxBase::DeviceManager::GetInstance()->DestroyDevices();
    if (ret != APP_ERR_OK) {
        LogError << "DestroyDevices failed";
        return ret;
    }
    LogInfo << "ending3...";
    return 0;
}