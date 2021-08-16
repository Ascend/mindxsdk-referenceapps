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

struct ReasonerConfig {
    uint32_t deviceId;
    uint32_t baseVideoChannelId;
    uint32_t maxTryOpenVideoStream;
    uint32_t maxDecodeFrameQueueLength;
    uint32_t popDecodeFrameWaitTime;
    uint32_t intervalPerformanceMonitorPrint;
    uint32_t intervalMainThreadControlCheck;
    uint32_t yoloModelWidth;
    uint32_t yoloModelHeight;
    std::string yoloModelPath;
    std::string yoloLabelPath;
    std::vector<std::string> rtspList;
    bool printDetectResult = true;
    bool writeDetectResultToFile = false;
    bool enablePerformanceMonitorPrint = true;
};

class MultiChannelVideoReasoner {
public:
    MultiChannelVideoReasoner() = default;
    ~MultiChannelVideoReasoner() = default;

    APP_ERROR Init(const ReasonerConfig &initConfig);
    APP_ERROR DeInit();
    void Process();

private:
    static void GetDecodeVideoFrame(const std::shared_ptr<AscendStreamPuller::StreamPuller> &streamPuller,
                                    const std::shared_ptr<AscendVideoDecoder::VideoDecoder> &videoDecoder,
                                    const int &index,
                                    const std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> &decodeFrameQueue,
                                    const MultiChannelVideoReasoner* multiChannelVideoReasoner);
    static void GetMultiChannelDetectionResult(const uint32_t &modelWidth, const uint32_t &modelHeight,
                                               const uint32_t &popDecodeFrameWaitTime,
                                               const MultiChannelVideoReasoner* multiChannelVideoReasoner);

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

public:
    static bool _s_force_stop;

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

private:
    std::vector<AscendStreamPuller::VideoFrameInfo> videoFrameInfos;
    std::vector<std::shared_ptr<AscendStreamPuller::StreamPuller>> streamPullers;
    std::vector<std::shared_ptr<AscendVideoDecoder::VideoDecoder>> videoDecoders;

    std::shared_ptr<AscendImageResizer::ImageResizer> imageResizer;
    std::shared_ptr<AscendYoloDetector::YoloDetector> yoloDetector;
    std::shared_ptr<AscendPerformanceMonitor::PerformanceMonitor> performanceMonitor;

    std::map<int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> decodeFrameQueueMap;

};

#endif //MULTICHANNELVIDEODETECTION_MULTICHANNELVIDEOREASONER_H
