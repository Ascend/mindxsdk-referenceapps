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

/**
 * Init MultiChannelVideoReasoner by {@link ReasonerConfig}
 * @param initConfig const reference to initial config
 * @return status code of whether initialization is successful
 */
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

    // init class member variable
    this->deviceId = initConfig.deviceId;
    this->yoloModelWidth = initConfig.yoloModelWidth;
    this->yoloModelHeight = initConfig.yoloModelHeight;
    this->popDecodeFrameWaitTime = initConfig.popDecodeFrameWaitTime;
    this->maxDecodeFrameQueueLength = initConfig.maxDecodeFrameQueueLength;
    this->intervalPerformanceMonitorPrint = initConfig.intervalPerformanceMonitorPrint;
    this->intervalMainThreadControlCheck = initConfig.intervalMainThreadControlCheck;
    this->printDetectResult = initConfig.printDetectResult;
    this->writeDetectResultToFile = initConfig.writeDetectResultToFile;

    this->stopFlag = false;
    return APP_ERR_OK;
}

/**
 * The work flow of MultiChannelVideoReasoner
 */
void MultiChannelVideoReasoner::Process()
{
    auto startTime = std::chrono::high_resolution_clock::now();

    // start threads (video pull and decode | image resize and yolo detect | performance monitoring)
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

        // judge whether the video of all channels is pulled and decoded finish
        bool allVideoDataPulledAndDecoded = true;
        for (uint32_t i = 0; i < streamPullers.size(); i++) {
            if (!streamPullers[i]->stopFlag) {
                allVideoDataPulledAndDecoded = false;
                break;
            }
        }

        // judge whether the all data in decode frame queue is processed
        bool allVideoDataProcessed = !Util::IsExistDataInQueueMap(decodeFrameQueueMap);

        // all channels video data pull, decode, resize and detect finish
        // quit MultiChannelVideoReasoner
        // quit PerformanceMonitor
        if (allVideoDataPulledAndDecoded && allVideoDataProcessed) {
            LogInfo << "All channels' video data pull, decode, resize and detect complete, quit MultiChannelVideoReasoner.";
            stopFlag = true;

            performanceMonitor->stopFlag = true;
            LogInfo << "All processor exit, Stop PerformanceMonitor.";
        }

        // force stop case
        if (MultiChannelVideoReasoner::_s_force_stop) {
            LogInfo << "Force stop MultiChannelVideoReasoner.";
            stopFlag = true;
            // force stop StreamPullers
            for (uint32_t i = 0; i < streamPullers.size(); i++) {
                if (!streamPullers[i]->stopFlag) {
                    streamPullers[i]->stopFlag = true;
                    LogInfo << "Force stop StreamPuller " << i + 1;
                }
            }
            // force stop PerformanceMonitor
            performanceMonitor->stopFlag = true;
            LogInfo << "Force stop PerformanceMonitor.";
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

/**
 * De-init MultiChannelVideoReasoner
 * @return status code of whether de-initialization is successful
 */
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

/**
 * Get the decoded video frame
 * >> first step: pull video stream data by specific rtsp address
 * >> second step: decode video frame from pulled video frame data
 * @param streamPuller const reference to the pointer to StreamPuller
 * @param videoDecoder const reference to the pointer to VideoDecoder
 * @param index const reference to curr rtsp index
 * @param decodeFrameQueue const reference to the pointer to decode frame queue
 * @param multiChannelVideoReasoner const pointer to MultiChannelVideoReasoner
 */
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
            LogInfo << "quit video stream pull and decode, index: " << index;
            streamPuller->stopFlag = true;
            return;
        }

        if (streamPuller->stopFlag) {
            LogInfo << "no video frame to pull and all pulled video frame decoded. quit!";
            LogInfo << "Total decode video frame num: " << videoDecoder->GetTotalDecodeFrameNum();
            return;
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

/**
 * Get the detect result of all channels
 * @param modelWidth const reference to the input width of YoloDetector
 * @param modelHeight const reference to the input height of YoloDetector
 * @param popDecodeFrameWaitTime const reference to wait time when decode frame queue is empty
 * @param multiChannelVideoReasoner const pointer to MultiChannelVideoReasoner
 */
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

    // check result dir
    if (multiChannelVideoReasoner->writeDetectResultToFile) {
        Util::CheckAndCreateResultDir(multiChannelVideoReasoner->videoFrameInfos.size());
    }

    auto imageResizer = multiChannelVideoReasoner->imageResizer;
    auto yoloDetector = multiChannelVideoReasoner->yoloDetector;
    auto decodeFrameQueueMap = multiChannelVideoReasoner->decodeFrameQueueMap;
    auto videoFrameInfos = multiChannelVideoReasoner->videoFrameInfos;
    auto performanceMonitor = multiChannelVideoReasoner->performanceMonitor;

    while(true) {
        if (multiChannelVideoReasoner->stopFlag) {
            LogInfo << "quit video frame resize and detect.";
            return;
        }

        // check whether exist data in all decode frame queues
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

            // get detect result
            std::vector<MxBase::ObjectInfo> results;
            results = Util::GetDetectionResult(objInfos, rtspIndex, resizeFrame.frameId,
                                               multiChannelVideoReasoner->printDetectResult);
            if (results.empty()) {
                LogInfo << "rtsp " << rtspIndex
                << " frame " << resizeFrame.frameId
                << " no detect result.";
                continue;
            }

            // save detect result
            if (multiChannelVideoReasoner->writeDetectResultToFile) {
                ret = Util::SaveResult(decodeFrame, results, videoFrameInfos[rtspIndex],
                                       resizeFrame.frameId, rtspIndex);
                if (ret != APP_ERR_OK) {
                    LogError << "Save result failed, ret=" << ret << ".";
                    continue;
                }
            }
        }
    }
}

/// ========== private Method ========== ///

/**
 * Create StreamPullers and VideoDecoders by {@link ReasonerConfig}
 * @param config const reference to the config
 * @return status code of whether creations is successful
 */
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

/**
 * Create ImageResizer by {@link ReasonerConfig}
 * @param config const reference to the config
 * @return status code of whether creation is successful
 */
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

/**
 * Create YoloDetector by {@link ReasonerConfig}
 * @param config const reference to the config
 * @return status code of whether creation is successful
 */
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

/**
 * Create PerformanceMonitor by {@link ReasonerConfig}
 * @param config const reference to the config
 * @return status code of whether creation is successful
 */
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

/**
 * Destroy StreamPullers and VideoDecoders
 * @return status code of whether destroy is successful
 */
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

/**
 * Destroy ImageResizer
 * @return status code of whether destroy is successful
 */
APP_ERROR MultiChannelVideoReasoner::DestroyImageResizer()
{
    APP_ERROR ret = imageResizer->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "ImageResizer DeInit failed.";
        return ret;
    }
    return APP_ERR_OK;
}

/**
 * Destroy YoloDetector
 * @return status code of whether destroy is successful
 */
APP_ERROR MultiChannelVideoReasoner::DestroyYoloDetector()
{
    APP_ERROR ret = yoloDetector->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "YoloDetector DeInit failed.";
        return ret;
    }
    return APP_ERR_OK;
}

/**
 * Destroy PerformanceMonitor
 * @return status code of whether destroy is successful
 */
APP_ERROR MultiChannelVideoReasoner::DestroyPerformanceMonitor()
{
    APP_ERROR ret = performanceMonitor->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "PerformanceMonitor DeInit failed.";
        return ret;
    }
    return APP_ERR_OK;
}

/**
 * Clear decode frame queue map, video frame infos, StreamPuller and VideoDecoder vector
 */
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

