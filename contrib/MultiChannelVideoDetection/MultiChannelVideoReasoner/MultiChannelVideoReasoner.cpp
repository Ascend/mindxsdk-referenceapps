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

namespace {
    // performance monitor tag
    const std::string StreamPullerTag = "StreamPuller"; /* NOLINT */
    const std::string VideoDecoderTag = "VideoDecoder"; /* NOLINT */
    const std::string ImageResizerTag = "ImageResizer"; /* NOLINT */
    const std::string YoloDetectorTag = "YoloDetector"; /* NOLINT */
    const std::string YoloDetectorInferTag = YoloDetectorTag + " Infer"; /* NOLINT */
    const std::string YoloDetectorPostProcessTag = YoloDetectorTag + " PostProcess"; /* NOLINT */
    const std::string GetDetectResultTag = "GetDetectResult"; /* NOLINT */
    const std::string SaveDetectResultTag = "SaveDetectResult"; /* NOLINT */
}

// init static variable
bool MultiChannelVideoReasoner::_s_force_stop = false;

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
    this->enableIndependentThreadForEachDetectStep = initConfig.enableIndependentThreadForEachDetectStep;

    this->stopFlag = false;
    return APP_ERR_OK;
}

/**
 * The work flow of MultiChannelVideoReasoner
 */
void MultiChannelVideoReasoner::Process()
{
    auto startTime = std::chrono::high_resolution_clock::now();

    // start work threads
    std::vector<std::thread> videoProcessThreads;
    StartWorkThreads(videoProcessThreads);

    // work flow control
    while (!stopFlag) {
        // check work flow and try to exit
        TryQuitReasoner();

        // force stop case
        if (MultiChannelVideoReasoner::_s_force_stop) {
            ForceStopReasoner();
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
 * > first step: pull video stream data by specific rtsp address
 * > second step: decode video frame from pulled video frame data
 *
 * @param streamPuller const reference to the pointer to StreamPuller
 * @param videoDecoder const reference to the pointer to VideoDecoder
 * @param index const reference to curr rtsp index
 * @param decodeFrameQueue const reference to the pointer to decode frame queue
 * @param multiChannelVideoReasoner const pointer to MultiChannelVideoReasoner
 */
void MultiChannelVideoReasoner::GetDecodeVideoFrame
        (const std::shared_ptr<AscendStreamPuller::StreamPuller> &streamPuller,
         const std::shared_ptr<AscendVideoDecoder::VideoDecoder> &videoDecoder,
         const int &index,
         const std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> &decodeFrameQueue,
         const std::shared_ptr<MultiChannelVideoReasoner> &multiChannelVideoReasoner)
{
    // set device
    APP_ERROR ret = SetDevice(multiChannelVideoReasoner->deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "GetDecodeVideoFrame setDevice failed, code: " << ret << ".";
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
        performanceMonitor->Collect(StreamPullerTag + " " + std::to_string(index + 1), costMs);

        if (videoFrameData.size == 0) {
            LogDebug << "empty video frame, not need decode, continue!";
            continue;
        }

        // video frame decode
        startTime = std::chrono::high_resolution_clock::now();
        videoDecoder->Decode(videoFrameData, decodeFrameQueue);
        endTime = std::chrono::high_resolution_clock::now();
        costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        performanceMonitor->Collect(VideoDecoderTag + " " + std::to_string(index + 1), costMs);
    }
}

/**
 * Get the yolo inference result of all channels (insert result into yoloInferenceResultQueue)
 * > first step: get front data from decode frame queue
 * > second step: resize video frame to match model input size
 * > third step: input into the yolo model for inference
 * > fourth step: put yolo inference result into yoloInferenceResult queue
 *
 * @param multiChannelVideoReasoner const pointer to MultiChannelVideoReasoner
 */
void MultiChannelVideoReasoner::GetYoloInferenceResult
        (const std::shared_ptr<MultiChannelVideoReasoner> &multiChannelVideoReasoner)
{
    // set device
    APP_ERROR ret = SetDevice(multiChannelVideoReasoner->deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "GetYoloInferenceResult setDevice failed, code: " << ret << ".";
        return;
    }

    // config
    auto popDecodeFrameWaitTime = multiChannelVideoReasoner->popDecodeFrameWaitTime;
    auto writeDetectResultToFile = multiChannelVideoReasoner->writeDetectResultToFile;

    // data set
    auto decodeFrameQueueMap = multiChannelVideoReasoner->decodeFrameQueueMap;
    auto yoloInferenceResultQueue = multiChannelVideoReasoner->yoloInferenceResultQueue;

    while (true) {
        if (multiChannelVideoReasoner->stopFlag) {
            LogInfo << "quit video frame resize and yolo inference.";
            return;
        }

        std::_Rb_tree_const_iterator<std::pair<const int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>>> iter;
        for (iter = decodeFrameQueueMap.begin(); iter != decodeFrameQueueMap.end(); iter++) {
            auto rtspIndex = iter->first;
            auto decodeFrameQueue = iter->second;
            if (decodeFrameQueue->IsEmpty()) {
                continue;
            }

            // get decode frame data
            std::shared_ptr<void> data = nullptr;
            ret = decodeFrameQueue->Pop(data, popDecodeFrameWaitTime);
            if (ret != APP_ERR_OK) {
                LogError << rtspIndex << " DecodeFrameQueueMap pop failed.";
                continue;
            }
            auto decodeFrame = std::make_shared<MxBase::MemoryData>();
            decodeFrame = std::static_pointer_cast<MxBase::MemoryData>(data);

            // resize frame
            MxBase::DvppDataInfo resizeFrame = {};
            ret = multiChannelVideoReasoner->ResizeImage(decodeFrame, resizeFrame, rtspIndex);
            if (ret != APP_ERR_OK) {
                LogError << "rtsp " << rtspIndex <<  " frame " << resizeFrame.frameId <<
                " resize image failed, ret = " << ret << " " << GetError(ret);
                continue;
            }

            // yolo infer
            std::vector<MxBase::TensorBase> yoloOutputs = {};
            ret = multiChannelVideoReasoner->YoloInference(resizeFrame, yoloOutputs);
            if (ret != APP_ERR_OK) {
                LogError << "rtsp " << rtspIndex <<  " frame " << resizeFrame.frameId <<
                " yolo infer image failed, ret = " << ret << " " << GetError(ret) << ".";
                continue;
            }

            // put yolo inference result into queue
            YoloResultWrapper yoloInferenceResultWrapper = {};
            yoloInferenceResultWrapper.rtspIndex = rtspIndex;
            yoloInferenceResultWrapper.frameId = resizeFrame.frameId;
            yoloInferenceResultWrapper.yoloOutputs = yoloOutputs;

            // only need to save result, assign it
            if (writeDetectResultToFile) {
                yoloInferenceResultWrapper.videoFrame = decodeFrame;
            }
            yoloInferenceResultQueue->Push(yoloInferenceResultWrapper, true);
        }
    }
}

/**
 * Get the yolo post process result (insert result into yoloPostProcessQueue)
 * > first step: get front data from decode frame yolo inference queue
 * > second step: input into the yolo model for post process
 * > third step: yolo post process result into yoloPostProcessResult queue
 *
 * @param multiChannelVideoReasoner const pointer to MultiChannelVideoReasoner
 */
void MultiChannelVideoReasoner::GetYoloPostProcessResult
        (const std::shared_ptr<MultiChannelVideoReasoner> &multiChannelVideoReasoner)
{
    // set device
    APP_ERROR ret = SetDevice(multiChannelVideoReasoner->deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "GetYoloPostProcessResult setDevice failed, code: " << ret << ".";
        return;
    }

    // config
    auto popDecodeFrameWaitTime = multiChannelVideoReasoner->popDecodeFrameWaitTime;

    // data set
    auto yoloInferenceResultQueue = multiChannelVideoReasoner->yoloInferenceResultQueue;
    auto yoloPostProcessResultQueue = multiChannelVideoReasoner->yoloPostProcessResultQueue;

    while (true) {
        if (multiChannelVideoReasoner->stopFlag) {
            LogInfo << "quit yolo post process.";
            return;
        }

        if (yoloInferenceResultQueue->IsEmpty()) {
            continue;
        }

        // get yolo inference result from queue
        YoloResultWrapper result;
        ret = yoloInferenceResultQueue->Pop(result, popDecodeFrameWaitTime);
        if (ret != APP_ERR_OK) {
            LogError << "YoloInferenceResultQueue pop failed.";
            continue;
        }

        // yolo post process
        ret = multiChannelVideoReasoner->YoloPostProcess(result);
        if (ret != APP_ERR_OK) {
            LogError << "rtsp " << result.rtspIndex <<  " frame " << result.frameId <<
            " yolo post process failed, ret = " << ret << " " << GetError(ret) << ".";
            continue;
        }

        // put post process result into queue
        yoloPostProcessResultQueue->Push(result, true);
    }
}

/**
 * Get and save(opt) the final detect result of video frame
 * > first step: get front data from decode frame yolo post process queue
 * > second step: get the final detect result from post process result
 * > third step(opt): save the detect result to file (frameId.jpg)
 *
 * @param multiChannelVideoReasoner const pointer to MultiChannelVideoReasoner
 */
void MultiChannelVideoReasoner::GetAndSaveDetectResult
        (const std::shared_ptr<MultiChannelVideoReasoner> &multiChannelVideoReasoner)
{
    // set device
    APP_ERROR ret = SetDevice(multiChannelVideoReasoner->deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "GetAndSaveDetectResult setDevice failed, code: " << ret << ".";
        return;
    }

    // config
    auto writeDetectResultToFile = multiChannelVideoReasoner->writeDetectResultToFile;
    auto popDecodeFrameWaitTime = multiChannelVideoReasoner->popDecodeFrameWaitTime;

    // data set
    auto yoloPostProcessResultQueue = multiChannelVideoReasoner->yoloPostProcessResultQueue;

    // check result dir
    if (writeDetectResultToFile) {
        Util::CheckAndCreateResultDir(multiChannelVideoReasoner->videoFrameInfos.size());
    }

    while (true) {
        if (multiChannelVideoReasoner->stopFlag) {
            LogInfo << "quit get and save detect result.";
            return;
        }

        if (yoloPostProcessResultQueue->IsEmpty()) {
            continue;
        }

        // get yolo post process result from queue
        YoloResultWrapper result;
        ret = yoloPostProcessResultQueue->Pop(result, popDecodeFrameWaitTime);
        if (ret != APP_ERR_OK) {
            LogError << "YoloPostProcessResultQueue pop failed.";
            continue;
        }

        // get detect result
        std::vector<MxBase::ObjectInfo> results;
        results = multiChannelVideoReasoner->GetDetectResult(result);
        if (results.empty()) {
            LogInfo << "rtsp " << result.rtspIndex << " frame " << result.frameId
                    << " no detect result.";
            continue;
        }

        // save detect result
        if (writeDetectResultToFile) {
            ret = multiChannelVideoReasoner->SaveDetectResult(result, results);
            if (ret != APP_ERR_OK) {
                LogError << "Save result failed, ret=" << ret << ".";
                continue;
            }
        }
    }
}

/**
 * Get the detect result of all channels
 * > first step: get front data from decode frame queue
 * > second step: resize video frame to match model input size
 * > third step: input into the yolo model for inference and post process
 * > fourth step: get the final detect result from post process result
 * > fifth step(opt): save the detect result to file (frameId.jpg)
 *
 * @param modelWidth const reference to the input width of YoloDetector
 * @param modelHeight const reference to the input height of YoloDetector
 * @param popDecodeFrameWaitTime const reference to wait time when decode frame queue is empty
 * @param multiChannelVideoReasoner const pointer to MultiChannelVideoReasoner
 */
void MultiChannelVideoReasoner::GetMultiChannelDetectionResult
        (const std::shared_ptr<MultiChannelVideoReasoner> &multiChannelVideoReasoner)
{
    // set device
    APP_ERROR ret = SetDevice(multiChannelVideoReasoner->deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "GetMultiChannelDetectionResult setDevice failed, code: " << ret << ".";
        return;
    }

    // config
    auto writeDetectResultToFile = multiChannelVideoReasoner->writeDetectResultToFile;
    auto popDecodeFrameWaitTime = multiChannelVideoReasoner->popDecodeFrameWaitTime;

    // data set
    auto decodeFrameQueueMap = multiChannelVideoReasoner->decodeFrameQueueMap;

    // check result dir
    if (writeDetectResultToFile) {
        Util::CheckAndCreateResultDir(multiChannelVideoReasoner->videoFrameInfos.size());
    }

    while (true) {
        if (multiChannelVideoReasoner->stopFlag) {
            LogInfo << "quit video frame resize and detect.";
            return;
        }

        std::_Rb_tree_const_iterator<std::pair<const int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>>> iter;
        for (iter = decodeFrameQueueMap.begin(); iter != decodeFrameQueueMap.end(); iter++) {
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
            ret = multiChannelVideoReasoner->ResizeImage(decodeFrame, resizeFrame, rtspIndex);
            if (ret != APP_ERR_OK) {
                LogError << "rtsp " << rtspIndex <<  " frame " << resizeFrame.frameId
                         << " resize image failed, ret = " << ret << " " << GetError(ret);
                continue;
            }

            // yolo detect
            YoloResultWrapper result;
            result.rtspIndex = rtspIndex;
            result.frameId = resizeFrame.frameId;
            ret = multiChannelVideoReasoner->YoloDetect(resizeFrame, result);
            if (ret != APP_ERR_OK) {
                LogError << "Yolo detect image failed, ret = " << ret << " " << GetError(ret) << ".";
                continue;
            }

            // get detect result
            std::vector<MxBase::ObjectInfo> results;
            results = multiChannelVideoReasoner->GetDetectResult(result);
            if (results.empty()) {
                LogInfo << "rtsp " << result.rtspIndex << " frame " << result.frameId
                        << " no detect result.";
                continue;
            }

            // save detect result
            if (writeDetectResultToFile) {
                result.videoFrame = decodeFrame;
                ret = multiChannelVideoReasoner->SaveDetectResult(result, results);
                if (ret != APP_ERR_OK) {
                    LogError << "Save result failed, ret=" << ret << ".";
                    continue;
                }
            }
        }
    }
}

/**
 * Set device
 * @param deviceId device id of need setting
 * @return status code of whether setting is successful
 */
APP_ERROR MultiChannelVideoReasoner::SetDevice(uint32_t deviceId)
{
    // set device
    MxBase::DeviceContext device;
    device.devId = (int32_t)deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->SetDevice(device);
    if (ret != APP_ERR_OK) {
        LogError << "SetDevice failed";
        return ret;
    }

    return APP_ERR_OK;
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
            LogError << "Init " << (i + 1) << " StreamPuller failed, stream name: " << rtspList[i] << ".";
            return ret;
        }
        videoFrameInfo = streamPuller->GetFrameInfo();

        Util::InitVideoDecoderParam(decoderInitParam, config.deviceId, config.baseVideoChannelId + i, videoFrameInfo);
        ret = videoDecoder->Init(decoderInitParam);
        if (ret != APP_ERR_OK) {
            LogError << "Init " << (i + 1) << " VideoDecoder failed.";
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
        LogError << "Init image resizer failed.";
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
    std::vector<std::string> objects;

    // add StreamPuller and VideoDecoder performance tag
    uint32_t size = config.rtspList.size();
    for (uint32_t i = 0; i < size; i++) {
        objects.push_back(StreamPullerTag + " " + std::to_string(i + 1));
        objects.push_back(VideoDecoderTag + " " + std::to_string(i + 1));
    }

    // add ImageResizer performance tag
    objects.emplace_back(ImageResizerTag);

    // add YoloDetector performance tag
    if (config.enableIndependentThreadForEachDetectStep) {
        objects.emplace_back(YoloDetectorInferTag);
        objects.emplace_back(YoloDetectorPostProcessTag);
    } else {
        objects.emplace_back(YoloDetectorTag);
    }

    // add get detect result performance tag
    objects.emplace_back(GetDetectResultTag);

    // add save detect result performance tag
    if (config.writeDetectResultToFile) {
        objects.emplace_back(SaveDetectResultTag);
    }

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
            LogError << "Deinit " << (i + 1) << " VideoDecoder failed.";
            return ret;
        }

        // deinit stream puller
        ret = streamPullers[i]->DeInit();
        if (ret != APP_ERR_OK) {
            LogError << "Deinit " << (i + 1) << " StreamPuller failed.";
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
void MultiChannelVideoReasoner::ClearData()
{
    LogInfo << "ClearData start.";

    // stop and clear queue
    Util::StopAndClearQueueMap(decodeFrameQueueMap);
    decodeFrameQueueMap.clear();
    if (enableIndependentThreadForEachDetectStep) {
        yoloInferenceResultQueue->Stop();
        yoloInferenceResultQueue->Clear();
        yoloPostProcessResultQueue->Stop();
        yoloPostProcessResultQueue->Clear();
    }

    videoFrameInfos.clear();
    videoDecoders.clear();
    streamPullers.clear();

    LogInfo << "ClearData successful.";
}

/**
 * Start work threads of multi-channel video reasoner and put them into set
 * > Thread Type-1: video stream pull and decode
 * > Thread Type-2: image resize and yolo infer
 * > Thread Type-3: yolo post process
 * > Thread Type-4: get and save final detect result
 * > Thread Type-5: yolo detect (integration steps 2-4)
 * > Thread Type-5: performance monitor
 *
 * @param workThreads reference to work threads set
 */
void MultiChannelVideoReasoner::StartWorkThreads(std::vector<std::thread> &workThreads)
{
    LogInfo << "Start work threads...";

    // specify an empty deleter to avoid double free
    auto deleter = [] (MultiChannelVideoReasoner *multiChannelVideoReasoner) {
    };
    auto multiChannelVideoReasoner = std::shared_ptr<MultiChannelVideoReasoner>(this, deleter);

    // start video pull and decode threads
    for (uint32_t i = 0; i < streamPullers.size(); i++) {
        auto decodeFrameQueue = std::make_shared<BlockingQueue<std::shared_ptr<void>>>(maxDecodeFrameQueueLength);
        std::thread getDecodeVideoFrame(GetDecodeVideoFrame,
                                        streamPullers[i], videoDecoders[i], i,
                                        decodeFrameQueue, multiChannelVideoReasoner);
        LogInfo << "video stream pull and decode thread " << (i + 1) << " start.";

        // save
        workThreads.push_back(std::move(getDecodeVideoFrame));
        decodeFrameQueueMap.insert(std::pair<int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>>
                                               (i, decodeFrameQueue));
    }

    if (!enableIndependentThreadForEachDetectStep) {
        // start yolo detect and get, save detect result thread
        std::thread getMultiChannelDetectionResult(GetMultiChannelDetectionResult, multiChannelVideoReasoner);
        workThreads.push_back(std::move(getMultiChannelDetectionResult));
        LogInfo << "yolo detect thread start.";
    } else {
        auto yoloProcessResultQueueSize = streamPullers.size() * maxDecodeFrameQueueLength;
        // start yolo infer thread
        yoloInferenceResultQueue = std::make_shared<BlockingQueue<YoloResultWrapper>>(yoloProcessResultQueueSize);
        std::thread getYoloInferenceResult(GetYoloInferenceResult, multiChannelVideoReasoner);
        workThreads.push_back(std::move(getYoloInferenceResult));
        LogInfo << "yolo infer thread start.";

        // start yolo post process thread
        yoloPostProcessResultQueue = std::make_shared<BlockingQueue<YoloResultWrapper>>(yoloProcessResultQueueSize);
        std::thread getYoloPostProcessResult(GetYoloPostProcessResult, multiChannelVideoReasoner);
        workThreads.push_back(std::move(getYoloPostProcessResult));
        LogInfo << "yolo post process thread start.";

        // start get and save detect result thread
        std::thread getAndSaveDetectResult(GetAndSaveDetectResult, multiChannelVideoReasoner);
        workThreads.push_back(std::move(getAndSaveDetectResult));
        LogInfo << "get and save detect result thread start.";
    }

    // start performance monitor thread
    std::thread monitorPerformance(performanceMonitor->PrintStatistics,
                                   performanceMonitor, intervalPerformanceMonitorPrint);
    workThreads.push_back(std::move(monitorPerformance));
    LogInfo << "performance monitor thread start.";
}

/**
 * check video process work flow and try to exit
 */
void MultiChannelVideoReasoner::TryQuitReasoner()
{
    // judge whether the video of all channels is pulled and decoded finish
    bool allVideoDataPulledAndDecoded = true;
    for (auto & streamPuller : streamPullers) {
        if (!streamPuller->stopFlag) {
            allVideoDataPulledAndDecoded = false;
            break;
        }
    }

    // judge whether the all data in decode frame queue is processed
    bool allVideoDataProcessed = !Util::IsExistDataInQueueMap(decodeFrameQueueMap);
    if (enableIndependentThreadForEachDetectStep) {
        if (!yoloInferenceResultQueue->IsEmpty() ||
            !yoloPostProcessResultQueue->IsEmpty()) {
            allVideoDataProcessed = false;
        }
    }

    // judge whether the all detect result all write complete
    bool allDetectResultWrote = true;
    if (writeDetectResultToFile && enableIndependentThreadForEachDetectStep) {
        if (!yoloInferenceResultQueue->IsEmpty() ||
            !yoloPostProcessResultQueue->IsEmpty()) {
            allDetectResultWrote = false;
        }
    }

    // all channels video data pull, decode, resize, detect and save(opt) finish
    // quit MultiChannelVideoReasoner
    // quit PerformanceMonitor
    if (allVideoDataPulledAndDecoded && allVideoDataProcessed && allDetectResultWrote) {
        stopFlag = true;
        std::string msg = writeDetectResultToFile ? " and write complete, " : " complete, ";
        LogInfo << "All channels' video data pull, decode, resize, detect" << msg
        << "quit MultiChannelVideoReasoner.";

        performanceMonitor->stopFlag = true;
        LogInfo << "All processor exit, Stop PerformanceMonitor.";
    }
}

/**
 * Force stop reasoner and its components
 */
void MultiChannelVideoReasoner::ForceStopReasoner()
{
    LogInfo << "Force stop MultiChannelVideoReasoner.";
    stopFlag = true;
    // force stop PerformanceMonitor
    performanceMonitor->stopFlag = true;
    LogInfo << "Force stop PerformanceMonitor.";

    // force stop and clear decode frame queues
    Util::StopAndClearQueueMap(decodeFrameQueueMap);
    // force stop and clear yoloInferenceResultQueue and yoloPostProcessResultQueue
    if (enableIndependentThreadForEachDetectStep) {
        yoloInferenceResultQueue->Stop();
        yoloInferenceResultQueue->Clear();
        yoloPostProcessResultQueue->Stop();
        yoloPostProcessResultQueue->Clear();
    }
}

/// === package method with performance monitor === ///

/**
 * package ImagResizer.ResizeFromMemory with performance
 * @param decodeFrame const reference to decode video frame
 * @param resizeFrame reference to resize video frame
 * @param rtspIndex curr rtsp video stream index
 * @return status code of whether resize is successful
 */
APP_ERROR MultiChannelVideoReasoner::ResizeImage(const std::shared_ptr<MxBase::MemoryData> &decodeFrame,
                                                 MxBase::DvppDataInfo &resizeFrame, uint32_t rtspIndex)
{
    // resize config
    AscendImageResizer::ResizeConfig resizeConfig = {};
    resizeConfig.originWidth = videoFrameInfos[rtspIndex].width;
    resizeConfig.originHeight = videoFrameInfos[rtspIndex].height;
    resizeConfig.resizeWidth = yoloModelWidth;
    resizeConfig.resizeHeight = yoloModelHeight;

    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = imageResizer->ResizeFromMemory(*decodeFrame, resizeConfig, resizeFrame);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    performanceMonitor->Collect(ImageResizerTag, costMs);

    return APP_ERR_OK;
}

/**
 * package YoloDetector.Inference with performance
 * @param resizeFrame const reference to resize video frame
 * @param yoloOutputs reference to yolo infer tensors
 * @return status code of whether yolo infer is successful
 */
APP_ERROR MultiChannelVideoReasoner::YoloInference(const MxBase::DvppDataInfo &resizeFrame,
                                                   std::vector<MxBase::TensorBase> &yoloOutputs)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = yoloDetector->Inference(resizeFrame, yoloOutputs);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    performanceMonitor->Collect(YoloDetectorInferTag, costMs);

    return APP_ERR_OK;
}

/**
 * package YoloDetector.PostProcess with performance
 * @param result reference to yolo result wrapper
 * @return status code of whether yolo postprocess is successful
 */
APP_ERROR MultiChannelVideoReasoner::YoloPostProcess(YoloResultWrapper &result)
{
    // post process config
    AscendYoloDetector::PostProcessConfig postProcessConfig = {};
    postProcessConfig.originWidth = videoFrameInfos[result.rtspIndex].width;
    postProcessConfig.originHeight = videoFrameInfos[result.rtspIndex].height;
    postProcessConfig.modelWidth = yoloModelWidth;
    postProcessConfig.modelHeight = yoloModelHeight;

    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = yoloDetector->PostProcess(result.yoloOutputs, postProcessConfig, result.yoloObjInfos);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    auto costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    performanceMonitor->Collect(YoloDetectorPostProcessTag, costMs);

    return APP_ERR_OK;
}

/**
 * package YoloDetector.Detect with performance
 * @param resizeFrame const reference to resize video frame
 * @param result reference to yolo result wrapper
 * @return status code of whether yolo detect is successful
 */
APP_ERROR MultiChannelVideoReasoner::YoloDetect(const MxBase::DvppDataInfo &resizeFrame, YoloResultWrapper &result)
{
    // post process config
    AscendYoloDetector::PostProcessConfig postProcessConfig = {};
    postProcessConfig.originWidth = videoFrameInfos[result.rtspIndex].width;
    postProcessConfig.originHeight = videoFrameInfos[result.rtspIndex].height;
    postProcessConfig.modelWidth = yoloModelWidth;
    postProcessConfig.modelHeight = yoloModelHeight;

    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = yoloDetector->Detect(resizeFrame, postProcessConfig, result.yoloObjInfos);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    auto costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    performanceMonitor->Collect(YoloDetectorTag, costMs);

    return APP_ERR_OK;
}

/**
 * package Util.GetDetectionResult with performance
 * @param result const reference to yolo result wrapper
 * @return detect result vector
 */
std::vector<MxBase::ObjectInfo> MultiChannelVideoReasoner::GetDetectResult(const YoloResultWrapper &result)
{
    std::vector<MxBase::ObjectInfo> results;
    auto startTime = std::chrono::high_resolution_clock::now();
    results = Util::GetDetectionResult(result.yoloObjInfos, result.rtspIndex,
                                       result.frameId, printDetectResult);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    performanceMonitor->Collect(GetDetectResultTag, costMs);

    return results;
}

/**
 * package Util.SaveResult with performance
 * @param result const reference to yolo result wrapper
 * @param results const reference to detect results
 * @return  status code of whether save result is successful
 */
APP_ERROR MultiChannelVideoReasoner::SaveDetectResult(const YoloResultWrapper &result,
                                                      const std::vector<MxBase::ObjectInfo> &results)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = Util::SaveResult(result.videoFrame, results,
                           videoFrameInfos[result.rtspIndex],
                           result.frameId, result.rtspIndex);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    auto costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    performanceMonitor->Collect(SaveDetectResultTag, costMs);

    return APP_ERR_OK;
}