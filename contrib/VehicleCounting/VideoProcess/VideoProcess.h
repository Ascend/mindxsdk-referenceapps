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
#ifndef STREAM_PULL_SAMPLE_VIDEOPROCESS_H
#define STREAM_PULL_SAMPLE_VIDEOPROCESS_H

#include "MxBase/ErrorCode/ErrorCodes.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
#include "ObjectPostProcessors/Yolov3PostProcess.h"
#include "../BlockingQueue/BlockingQueue.h"
#include "../Yolov4Detection/Yolov4Detection.h"
#include "MOTConnection.h"

struct center{
    double x;
    double y;
};

class VideoProcess {
public:
    VideoProcess();
    std::queue<cv::Mat> Getframes();
    APP_ERROR StreamInit(const std::string &rtspUrl);
    APP_ERROR StreamDeInit();
    APP_ERROR VideoDecodeInit();
    APP_ERROR VideoDecodeDeInit();
    static void GetFrames(std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>  blockingQueue, 
	                      std::shared_ptr<VideoProcess> videoProcess);
    static void GetResults(std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> blockingQueue, 
	                       std::shared_ptr<Yolov4Detection> yolov4Detection,
	                       std::shared_ptr<VideoProcess> videoProcess, std::shared_ptr<ascendVehicleTracking::MOTConnection> tracker);
private:
    static APP_ERROR VideoDecodeCallback(std::shared_ptr<void> buffer, 
	                                    MxBase::DvppDataInfo &inputDataInfo, void *userData);
    APP_ERROR VideoDecode(MxBase::MemoryData &streamData, const uint32_t &height, 
	                    const uint32_t &width, void *userData);
    APP_ERROR SaveResult(const std::shared_ptr<MxBase::MemoryData> resulInfo, const uint32_t frameId,
                         std::vector<MxBase::ObjectInfo> &objInfos_);
public:
    static bool stopFlag;
    static const uint32_t DEVICE_ID = 0;
private:
    std::queue<cv::Mat> frameIf;
    int color_num[200][3]; // 随机颜色存储
    std::shared_ptr<MxBase::DvppWrapper> vDvppWrapper;
    const uint32_t CHANNEL_ID = 0;
};
#endif //STREAM_PULL_SAMPLE_VIDEOPROCESS_H