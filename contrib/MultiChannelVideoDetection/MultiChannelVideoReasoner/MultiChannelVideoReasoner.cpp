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

#include "MultiChannelVideoReasoner.h"
#include "Util/Util.h"

#include "MxBase/DeviceManager/DeviceManager.h"

#include <thread>

APP_ERROR MultiChannelVideoReasoner::Init(const ReasonerConfig &initConfig)
{
    LogDebug << "Init MultiChannelVideoReasoner start.";
    APP_ERROR ret;

    ret = CreateStreamPullerAndVideoDecoder(initConfig);
    if (ret != APP_ERR_OK) {
        LogError << "CreateStreamPullerAndVideoDecoder failed.";
        return ret;
    }

    ret = CreateImageResizer(initConfig);
    if (ret != APP_ERR_OK) {
        LogError << "CreateImageResizer failed.";
        return ret;
    }

    ret = CreateYoloDetector(initConfig);
    if (ret != APP_ERR_OK) {
        LogError << "CreateYoloDetector failed.";
        return ret;
    }

    ret = CreatePerformanceMonitor(initConfig);
    if (ret != APP_ERR_OK) {
        LogError << "CreatePerformanceMonitor failed.";
        return ret;
    }

    this->deviceId = initConfig.deviceId;
    this->yoloModelWidth = initConfig.yoloModelWidth;
    this->yoloModelHeight = initConfig.yoloModelHeight;
    this->popDecodeFrameWaitTime = initConfig.popDecodeFrameWaitTime;
    this->maxDecodeFrameQueueLength = initConfig.maxDecodeFrameQueueLength;
    this->intervalPerformanceMonitorPrint = initConfig.intervalPerformanceMonitorPrint;
    this->intervalMainThreadControlCheck = initConfig.intervalMainThreadControlCheck;
    this->writeDetectResultToFile = initConfig.writeDetectResultToFile;

    this->stopFlag = false;
    return APP_ERR_OK;
}

void MultiChannelVideoReasoner::Process()
{
    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> videoProcessThreads;
    for (uint32_t i = 0; i < streamPullers.size(); i++) {
        auto decodeFrameQueue = std::make_shared<BlockingQueue<std::shared_ptr<void>>>(maxDecodeFrameQueueLength);
        std::thread getDecodeVideoFrame(GetDecodeVideoFrame,
                                        streamPullers[i], videoDecoders[i], i, decodeFrameQueue, this);

        // save
        videoProcessThreads.push_back(std::move(getDecodeVideoFrame));
        decodeFrameQueueMap.insert(std::pair<int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>>
                                                (i, decodeFrameQueue));
    }

    std::thread getMultiChannelDetectionResult(GetMultiChannelDetectionResult,
                                               yoloModelWidth, yoloModelHeight, popDecodeFrameWaitTime, this);
    videoProcessThreads.push_back(std::move(getMultiChannelDetectionResult));
    std::thread monitorPerformance(performanceMonitor->PrintStatistics,
                                   performanceMonitor, intervalPerformanceMonitorPrint);
    videoProcessThreads.push_back(std::move(monitorPerformance));

    while (!stopFlag) {
        bool allVideoDataPulledAndDecoded = true;

        for (uint32_t i = 0; i < streamPullers.size(); i++) {
            if (streamPullers[i]->stopFlag) {
                if (!videoDecoders[i]->stopFlag) {
                    LogDebug << "all video frame decoded and no fresh data, quit video decoder "
                                + std::to_string(i) + ".";
                    videoDecoders[i]->stopFlag = true;
                }
            } else {
                allVideoDataPulledAndDecoded = false;
            }
        }

        bool allVideoDataProcessed = !Util::IsExistDataInQueueMap(decodeFrameQueueMap);

        if (allVideoDataPulledAndDecoded && allVideoDataProcessed) {
            LogDebug << "all decoded frame detected and no fresh data, quit image resizer and yolo detector";
            imageResizer->stopFlag = true;
            yoloDetector->stopFlag = true;
        }

        if (allVideoDataPulledAndDecoded && imageResizer->stopFlag && yoloDetector->stopFlag) {
            LogDebug << "Both of stream puller, video decoder, image resizer and yolo detector quit, main quit";
            stopFlag = true;
        }

        // force stop case
        if (MultiChannelVideoReasoner::_s_force_stop) {
            LogInfo << "Force stop MultiChannelVideoReasoner.";
            stopFlag = true;
        }

        if (stopFlag) {
            LogInfo << "All processor quit, quit performance monitor.";
            performanceMonitor->stopFlag = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(intervalMainThreadControlCheck));
    }

    // threads join
    for (auto & videoProcessThread : videoProcessThreads) {
        videoProcessThread.join();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double costS = std::chrono::duration<double>(endTime - startTime).count();
    LogInfo << "total process time: " << costS << "s.";
}

APP_ERROR MultiChannelVideoReasoner::DeInit()
{
    APP_ERROR ret;

    ret = DestroyStreamPullerAndVideoDecoder();
    if (ret != APP_ERR_OK) {
        LogError << "DestroyStreamPullerAndVideoDecoder failed.";
        return ret;
    }

    ret = DestroyImageResizer();
    if (ret != APP_ERR_OK) {
        LogError << "DestroyImageResizer failed.";
        return ret;
    }

    ret = DestroyYoloDetector();
    if (ret != APP_ERR_OK) {
        LogError << "DestroyYoloDetector failed.";
        return ret;
    }

    ret = DestroyPerformanceMonitor();
    if (ret != APP_ERR_OK) {
        LogError << "DestroyPerformanceMonitor failed.";
        return ret;
    }

    ClearData();

    this->stopFlag = true;
    return APP_ERR_OK;
}

/// ========== static Method ========== ///
void MultiChannelVideoReasoner::GetDecodeVideoFrame(
        const std::shared_ptr<AscendStreamPuller::StreamPuller> &streamPuller,
        const std::shared_ptr<AscendVideoDecoder::VideoDecoder> &videoDecoder,
        const int &index,
        const std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> &decodeFrameQueue,
        const MultiChannelVideoReasoner* multiChannelVideoReasoner)
{
    // set device
    MxBase::DeviceContext device;
    device.devId = (int32_t) multiChannelVideoReasoner->deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->SetDevice(device);
    if (ret != APP_ERR_OK) {
        LogError << "SetDevice failed";
        return;
    }

    auto performanceMonitor = multiChannelVideoReasoner->performanceMonitor;
    while (true) {
        if (multiChannelVideoReasoner->stopFlag) {
            LogDebug << "stop video stream pull and video frame decode";
            streamPuller->stopFlag = true;
            videoDecoder->stopFlag = true;
            break;
        }

        if (streamPuller->stopFlag) {
            LogDebug << "no video frame to pull and all pulled video frame decoded. quit!";
            LogInfo << "Total decode video frame num: " << videoDecoder->GetTotalDecodeFrameNum();
            break;
        }

        // video stream pull
        auto startTime = std::chrono::high_resolution_clock::now();
        auto videoFrameData = streamPuller->GetNextFrame();
        auto endTime = std::chrono::high_resolution_clock::now();
        double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        performanceMonitor->Collect("StreamPuller " + std::to_string(index + 1), costMs);

        if (videoFrameData.size == 0) {
            LogDebug << "empty video frame, not need decode, continue!";
            continue;
        }

        // video frame decode
        startTime = std::chrono::high_resolution_clock::now();
        videoDecoder->Decode(videoFrameData, decodeFrameQueue.get());
        endTime = std::chrono::high_resolution_clock::now();
        costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        performanceMonitor->Collect("VideoDecoder " + std::to_string(index + 1), costMs);
    }
}

void MultiChannelVideoReasoner::GetMultiChannelDetectionResult(
        const uint32_t &modelWidth,
        const uint32_t &modelHeight,
        const uint32_t &popDecodeFrameWaitTime,
        const MultiChannelVideoReasoner* multiChannelVideoReasoner)
{
    // set device
    MxBase::DeviceContext device;
    device.devId = (int32_t) multiChannelVideoReasoner->deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->SetDevice(device);
    if (ret != APP_ERR_OK) {
        LogError << "SetDevice failed";
        return;
    }

    auto imageResizer = multiChannelVideoReasoner->imageResizer;
    auto yoloDetector = multiChannelVideoReasoner->yoloDetector;
    auto decodeFrameQueueMap = multiChannelVideoReasoner->decodeFrameQueueMap;
    auto videoFrameInfos = multiChannelVideoReasoner->videoFrameInfos;
    auto performanceMonitor = multiChannelVideoReasoner->performanceMonitor;

    while(true) {
        if (multiChannelVideoReasoner->stopFlag) {
            LogDebug << "stop image resize and yolo detect";
            imageResizer->stopFlag = true;
            yoloDetector->stopFlag = true;
            break;
        }

        if (imageResizer->stopFlag && yoloDetector->stopFlag) {
            LogDebug << "no image need to resize and all image detected. quit!";
            break;
        }

        if (!Util::IsExistDataInQueueMap(decodeFrameQueueMap)) {
            continue;
        }

        std::_Rb_tree_const_iterator<std::pair<const int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>>> iter;
        for (iter = decodeFrameQueueMap.begin();iter != decodeFrameQueueMap.end();iter++) {
            auto rtspIndex = iter->first;
            auto decodeFrameQueue = iter->second;
            if (decodeFrameQueue->IsEmpty()) {
                continue;
            }

            // get decode frame data
            std::shared_ptr<void> data = nullptr;
            ret = decodeFrameQueue->Pop(data, popDecodeFrameWaitTime);
            if (ret != APP_ERR_OK) {
                LogError << "Pop failed";
                continue;
            }
            auto decodeFrame = std::make_shared<MxBase::MemoryData>();
            decodeFrame = std::static_pointer_cast<MxBase::MemoryData>(data);

            // resize frame
            MxBase::DvppDataInfo resizeFrame = {};
            auto startTime = std::chrono::high_resolution_clock::now();
            ret = imageResizer->ResizeFromMemory(*decodeFrame,
                                                 videoFrameInfos[rtspIndex].width, videoFrameInfos[rtspIndex].height,
                                                 modelWidth, modelHeight, resizeFrame);
            if (ret != APP_ERR_OK) {
                LogError << "Resize image failed, ret = " << ret << " " << GetError(ret);
                continue;
            }
            auto endTime = std::chrono::high_resolution_clock::now();
            double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
            performanceMonitor->Collect("ImageResizer", costMs);

            // yolo detect
            std::vector<std::vector<MxBase::ObjectInfo>> objInfos;
            startTime = std::chrono::high_resolution_clock::now();
            ret = yoloDetector->Detect(resizeFrame, objInfos,
                                       videoFrameInfos[rtspIndex].width, videoFrameInfos[rtspIndex].height,
                                       modelWidth, modelHeight);
            if (ret != APP_ERR_OK) {
                LogError << "Yolo detect image failed, ret = " << ret << " " << GetError(ret) << ".";
                continue;
            }
            endTime = std::chrono::high_resolution_clock::now();
            costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
            performanceMonitor->Collect("YoloDetector", costMs);

            // save detect result
            if (multiChannelVideoReasoner->writeDetectResultToFile) {
                ret = Util::SaveResult(decodeFrame, objInfos,
                                       resizeFrame.frameId,
                                       videoFrameInfos[rtspIndex].width, videoFrameInfos[rtspIndex].height,
                                       videoFrameInfos.size(), rtspIndex);
                if (ret != APP_ERR_OK) {
                    LogError << "Save result failed, ret=" << ret << ".";
                    return;
                }
            }
        }
    }
}

/// ========== private Method ========== ///
APP_ERROR MultiChannelVideoReasoner::CreateStreamPullerAndVideoDecoder(const ReasonerConfig &config)
{
    auto rtspList = config.rtspList;

    APP_ERROR ret;
    AscendStreamPuller::VideoFrameInfo videoFrameInfo;
    AscendVideoDecoder::DecoderInitParam decoderInitParam = {};

    for (uint32_t i = 0; i < rtspList.size(); i++) {
        auto streamPuller = std::make_shared<AscendStreamPuller::StreamPuller>();
        auto videoDecoder = std::make_shared<AscendVideoDecoder::VideoDecoder>();

        ret = streamPuller->Init(rtspList[i], config.maxTryOpenVideoStream, config.deviceId);
        if (ret != APP_ERR_OK) {
            LogError << "Init " << i + 1 << " StreamPuller failed, stream name: " << rtspList[i];
            return ret;
        }
        videoFrameInfo = streamPuller->GetFrameInfo();

        Util::InitVideoDecoderParam(decoderInitParam, config.deviceId, config.baseVideoChannelId + i, videoFrameInfo);
        ret = videoDecoder->Init(decoderInitParam);
        if (ret != APP_ERR_OK) {
            LogError << "Init " << i + 1 << " VideoDecoder failed";
            return ret;
        }

        // save
        streamPullers.push_back(streamPuller);
        videoDecoders.push_back(videoDecoder);
        videoFrameInfos.push_back(videoFrameInfo);
    }

    return APP_ERR_OK;
}

APP_ERROR MultiChannelVideoReasoner::CreateImageResizer(const ReasonerConfig &config)
{
    imageResizer = std::make_shared<AscendImageResizer::ImageResizer>();

    APP_ERROR ret = imageResizer->Init(config.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Init image resizer failed";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR MultiChannelVideoReasoner::CreateYoloDetector(const ReasonerConfig &config)
{
    yoloDetector = std::make_shared<AscendYoloDetector::YoloDetector>();

    AscendYoloDetector::YoloInitParam yoloInitParam;
    Util::InitYoloParam(yoloInitParam, config.deviceId, config.yoloLabelPath, config.yoloModelPath);
    APP_ERROR ret = yoloDetector->Init(yoloInitParam);
    if (ret != APP_ERR_OK) {
        LogError << "Init yolo detector failed.";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR MultiChannelVideoReasoner::CreatePerformanceMonitor(const ReasonerConfig &config)
{
    uint32_t size = config.rtspList.size();

    std::vector<std::string> objects;
    for (uint32_t i = 0; i < size; i++) {
        objects.push_back("StreamPuller " + std::to_string(i + 1));
        objects.push_back("VideoDecoder " + std::to_string(i + 1));
    }
    objects.emplace_back("ImageResizer");
    objects.emplace_back("YoloDetector");

    performanceMonitor = std::make_shared<AscendPerformanceMonitor::PerformanceMonitor>();
    APP_ERROR ret = performanceMonitor->Init(objects, config.enablePerformanceMonitorPrint);
    if (ret != APP_ERR_OK) {
        LogError << "Init performance monitor failed.";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR MultiChannelVideoReasoner::DestroyStreamPullerAndVideoDecoder()
{
    APP_ERROR ret;
    // deinit stream pullers and video decoders
    for (uint32_t i = 0; i < streamPullers.size(); i++) {
        // deinit video decoder
        ret = videoDecoders[i]->DeInit();
        if (ret != APP_ERR_OK) {
            LogError << "Deinit " << i + 1 << " VideoDecoder failed";
            return ret;
        }

        // deinit stream puller
        ret = streamPullers[i]->DeInit();
        if (ret != APP_ERR_OK) {
            LogError << "Deinit " << i + 1 << " StreamPuller failed.";
            return ret;
        }
    }

    return APP_ERR_OK;
}

APP_ERROR MultiChannelVideoReasoner::DestroyImageResizer()
{
    APP_ERROR ret = imageResizer->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "ImageResizer DeInit failed.";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR MultiChannelVideoReasoner::DestroyYoloDetector()
{
    APP_ERROR ret = yoloDetector->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "YoloDetector DeInit failed.";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR MultiChannelVideoReasoner::DestroyPerformanceMonitor()
{
    APP_ERROR ret = performanceMonitor->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "PerformanceMonitor DeInit failed.";
        return ret;
    }
    return APP_ERR_OK;
}

void MultiChannelVideoReasoner::ClearData() {
    // stop and clear queue
    std::_Rb_tree_const_iterator<std::pair<const int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>>> iter;
    for (iter = decodeFrameQueueMap.begin();iter != decodeFrameQueueMap.end();iter++) {
        iter->second->Stop();
        iter->second->Clear();
    }
    decodeFrameQueueMap.clear();

    videoFrameInfos.clear();
    videoDecoders.clear();
    streamPullers.clear();
}

