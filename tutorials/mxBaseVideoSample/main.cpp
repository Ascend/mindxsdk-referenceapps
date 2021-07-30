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
#include "BlockingQueue/BlockingQueue.h"
#include "VideoProcess/VideoProcess.h"
#include "Yolov3Detection/Yolov3Detection.h"

bool VideoProcess::stopFlag = false;
std::vector<double> g_inferCost;
namespace {
    const uint32_t MAX_QUEUE_LENGHT = 1000;
}

static void SigHandler(int signal)
{
    if (signal == SIGINT) {
        VideoProcess::stopFlag = true;
    }
}

void InitYolov3Param(InitParam &initParam, const uint32_t deviceID)
{
    initParam.deviceId = deviceID;
    initParam.labelPath = "./model/coco.names";
    initParam.checkTensor = true;
    initParam.modelPath = "{yolov3模型路径}";
    initParam.classNum = 80;
    initParam.biasesNum = 18;
    initParam.biases = "10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326";
    initParam.objectnessThresh = "0.001";
    initParam.iouThresh = "0.5";
    initParam.scoreThresh = "0.001";
    initParam.yoloType = 3;
    initParam.modelType = 0;
    initParam.inputType = 0;
    initParam.anchorDim = 3;
}

int main() {
    std::string streamName = "rtsp_Url";
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "InitDevices failed";
        return ret;
    }

    auto videoProcess = std::make_shared<VideoProcess>();
    auto yolov3 = std::make_shared<Yolov3Detection>();

    InitParam initParam;
    InitYolov3Param(initParam, videoProcess->DEVICE_ID);
    yolov3->FrameInit(initParam);
    MxBase::DeviceContext device;
    device.devId = videoProcess->DEVICE_ID;
    ret = MxBase::DeviceManager::GetInstance()->SetDevice(device);
    if (ret != APP_ERR_OK) {
        LogError << "SetDevice failed";
        return ret;
    }
    ret = videoProcess->StreamInit(streamName);
    if (ret != APP_ERR_OK) {
        LogError << "StreamInit failed";
        return ret;
    }
    ret = videoProcess->VideoDecodeInit();
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecodeInit failed";
        MxBase::DeviceManager::GetInstance()->DestroyDevices();
        return ret;
    }

    auto blockingQueue = std::make_shared<BlockingQueue<std::shared_ptr<void>>>(MAX_QUEUE_LENGHT);
    std::thread getFrame(videoProcess->GetFrames, blockingQueue, videoProcess);
    std::thread getResult(videoProcess->GetResults, blockingQueue, yolov3, videoProcess);

    if (signal(SIGINT, SigHandler) == SIG_ERR) {
        LogError << "can not catch SIGINT";
        return APP_ERR_COMM_FAILURE;
    }

    while (!videoProcess->stopFlag) {
        sleep(10);
    }
    getFrame.join();
    getResult.join();

    blockingQueue->Stop();
    blockingQueue->Clear();

    ret = yolov3->FrameDeInit();
    if (ret != APP_ERR_OK) {
        LogError << "FrameInit failed";
        return ret;
    }
    ret = videoProcess->StreamDeInit();
    if (ret != APP_ERR_OK) {
        LogError << "StreamDeInit failed";
        return ret;
    }
    ret = videoProcess->VideoDecodeDeInit();
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecodeDeInit failed";
        return ret;
    }
    ret = MxBase::DeviceManager::GetInstance()->DestroyDevices();
    if (ret != APP_ERR_OK) {
        LogError << "DestroyDevices failed";
        return ret;
    }
    return 0;
}