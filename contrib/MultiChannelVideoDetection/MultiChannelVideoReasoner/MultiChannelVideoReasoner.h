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

#ifndef MULTICHANNELVIDEODETECTION_MULTICHANNELVIDEOREASONER_H
#define MULTICHANNELVIDEODETECTION_MULTICHANNELVIDEOREASONER_H

#include "StreamPuller/StreamPuller.h"
#include "VideoDecoder/VideoDecoder.h"
#include "ImageResizer/ImageResizer.h"
#include "YoloDetector/YoloDetector.h"
#include "BlockingQueue/BlockingQueue.h"
#include "Util/PerformanceMonitor/PerformanceMonitor.h"

#include <thread>

// config
const uint32_t DEFAULT_CONTROL_CHECK_INTERVAL = 2;
const uint32_t DEFAULT_POP_WAIT_TIME = 10;

// when enable independentThread and writeDetectResult, please use this
const uint32_t DECODE_QUEUE_LENGTH_100 = 100;
// when enable independentThread, please use this
const uint32_t DECODE_QUEUE_LENGTH_200 = 200;
// when disable independentThread, please use this
const uint32_t DECODE_QUEUE_LENGTH_400 = 400;

struct ReasonerConfig {
    uint32_t deviceId;
    uint32_t baseVideoChannelId;
    // max times of retrying to open video Stream
    uint32_t maxTryOpenVideoStream;
    // max length of queue which cache decode video frame
    uint32_t maxDecodeFrameQueueLength;
    // wait time when decode frame queue is empty
    uint32_t popDecodeFrameWaitTime;
    // interval of printing performance message
    uint32_t intervalPerformanceMonitorPrint;
    // interval of main thread check work flow
    uint32_t intervalMainThreadControlCheck;
    // the input width of YoloDetector
    uint32_t yoloModelWidth;
    // the input height of YoloDetector
    uint32_t yoloModelHeight;
    // the path of yolo model
    std::string yoloModelPath;
    // the path of yolo model label
    std::string yoloLabelPath;
    // the rtsp video stream list which need to process
    std::vector<std::string> rtspList;
    // whether print yolo detect result, default: true
    bool printDetectResult = true;
    // whether write detect result to file, default: false
    bool writeDetectResultToFile = false;
    // whether print performance message, default: true
    bool enablePerformanceMonitorPrint = true;
    // whether enable independent thread for each detect step, default: true
    bool enableIndependentThreadForEachDetectStep = true;
};

struct YoloResultWrapper {
    uint32_t rtspIndex;
    uint32_t frameId;
    std::shared_ptr<MxBase::MemoryData> videoFrame;
    std::vector<MxBase::TensorBase> yoloOutputs;
    std::vector<std::vector<MxBase::ObjectInfo>> yoloObjInfos;
};

class MultiChannelVideoReasoner {
public:
    MultiChannelVideoReasoner() = default;
    ~MultiChannelVideoReasoner() = default;

    APP_ERROR Init(const ReasonerConfig &initConfig);
    APP_ERROR DeInit();
    void Process();

public:
    static bool _s_force_stop;

private:
    static void GetDecodeVideoFrame(const std::shared_ptr<AscendStreamPuller::StreamPuller> &streamPuller,
                                    const std::shared_ptr<AscendVideoDecoder::VideoDecoder> &videoDecoder,
                                    const int &index,
                                    const std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> &decodeFrameQueue,
                                    const std::shared_ptr<MultiChannelVideoReasoner> &multiChannelVideoReasoner);

    static void GetYoloInferenceResult(const std::shared_ptr<MultiChannelVideoReasoner> &multiChannelVideoReasoner);

    static void GetYoloPostProcessResult(const std::shared_ptr<MultiChannelVideoReasoner> &multiChannelVideoReasoner);

    static void GetAndSaveDetectResult(const std::shared_ptr<MultiChannelVideoReasoner> &multiChannelVideoReasoner);

    static void GetMultiChannelDetectionResult
            (const std::shared_ptr<MultiChannelVideoReasoner> &multiChannelVideoReasoner);

    static APP_ERROR SetDevice(uint32_t deviceId);

private:
    APP_ERROR CreateStreamPullerAndVideoDecoder(const ReasonerConfig &config);
    APP_ERROR CreateImageResizer(const ReasonerConfig &config);
    APP_ERROR CreateYoloDetector(const ReasonerConfig &config);
    APP_ERROR CreatePerformanceMonitor(const ReasonerConfig &config);

    APP_ERROR DestroyStreamPullerAndVideoDecoder();
    APP_ERROR DestroyImageResizer();
    APP_ERROR DestroyYoloDetector();
    APP_ERROR DestroyPerformanceMonitor();

    void ClearData();

    void StartWorkThreads(std::vector<std::thread>& workThreads);
    void TryQuitReasoner();
    void ForceStopReasoner();

private:
    uint32_t deviceId;
    bool stopFlag;

    uint32_t yoloModelWidth;
    uint32_t yoloModelHeight;
    uint32_t popDecodeFrameWaitTime;
    uint32_t maxDecodeFrameQueueLength;
    uint32_t intervalPerformanceMonitorPrint;
    uint32_t intervalMainThreadControlCheck;
    bool printDetectResult;
    bool writeDetectResultToFile;
    bool enableIndependentThreadForEachDetectStep;

private:
    std::vector<AscendStreamPuller::VideoFrameInfo> videoFrameInfos;
    std::vector<std::shared_ptr<AscendStreamPuller::StreamPuller>> streamPullers;
    std::vector<std::shared_ptr<AscendVideoDecoder::VideoDecoder>> videoDecoders;

    std::shared_ptr<AscendImageResizer::ImageResizer> imageResizer;
    std::shared_ptr<AscendYoloDetector::YoloDetector> yoloDetector;
    std::shared_ptr<AscendPerformanceMonitor::PerformanceMonitor> performanceMonitor;

    // queue which save the yolo inference result
    std::shared_ptr<BlockingQueue<YoloResultWrapper>> yoloInferenceResultQueue;
    // queue which save the yolo post process result
    std::shared_ptr<BlockingQueue<YoloResultWrapper>> yoloPostProcessResultQueue;
    // map which save the pointers to decode frame queue
    std::map<int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> decodeFrameQueueMap;
};
#endif // MULTICHANNELVIDEODETECTION_MULTICHANNELVIDEOREASONER_H
