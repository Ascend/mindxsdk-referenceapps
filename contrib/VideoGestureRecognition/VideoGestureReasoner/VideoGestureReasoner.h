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

#ifndef VIDEOGESTUREREASONER_VIDEOGESTUREREASONER_H
#define VIDEOGESTUREREASONER_VIDEOGESTUREREASONER_H

#include <thread>
#include "../FrameSkippingSampling/FrameSkippingSampling.h"
#include "../StreamPuller/StreamPuller.h"
#include "../VideoDecoder/VideoDecoder.h"
#include "../ImageResizer/ImageResizer.h"
#include "../ResnetDetector/ResnetDetector.h"

struct ReasonerConfig {
    uint32_t deviceId;
    uint32_t baseVideoChannelId;
    std::vector<std::string> rtspList;
    uint32_t maxTryOpenVideoStream;
    std::string resnetModelPath;
    std::string resnetLabelPath;
    uint32_t resnetModelWidth;
    uint32_t resnetModelHeight;
    uint32_t maxDecodeFrameQueueLength;
    uint32_t popDecodeFrameWaitTime;
    uint32_t SamplingInterval;
    uint32_t maxSamplingInterval;
};

class VideoGestureReasoner {
public:
    VideoGestureReasoner() = default;
    ~VideoGestureReasoner() = default;

    APP_ERROR Init(const ReasonerConfig &initConfig);
    APP_ERROR DeInit();
    void Process();

private:
    static void GetDecodeVideoFrame(const std::shared_ptr<AscendStreamPuller::StreamPuller> &streamPuller,
                                    const std::shared_ptr<AscendVideoDecoder::VideoDecoder> &videoDecoder,
                                    const std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> &decodeFrameQueue,
                                    const VideoGestureReasoner* videoGestureReasoner);
    static void GetDetectionResult(const uint32_t &modelWidth, const uint32_t &modelHeight,
                                               const uint32_t &popDecodeFrameWaitTime,
                                               const VideoGestureReasoner* videoGestureReasoner);

private:
    APP_ERROR CreateStreamPullerAndVideoDecoder(const ReasonerConfig &config);
    APP_ERROR CreateFrameSkippingSampling(const ReasonerConfig &config);
    APP_ERROR CreateImageResizer(const ReasonerConfig &config);
    APP_ERROR CreateResnetDetector(const ReasonerConfig &config);

    APP_ERROR DestroyStreamPullerAndVideoDecoder();
    APP_ERROR DestroyFrameSkippingSampling();
    APP_ERROR DestroyImageResizer();
    APP_ERROR DestroyResnetDetector();

    void ClearData();

public:
    static bool forceStop;

private:
    uint32_t deviceId;
    bool stopFlag;

    uint32_t resnetModelWidth;
    uint32_t resnetModelHeight;
    uint32_t popDecodeFrameWaitTime;
    uint32_t maxDecodeFrameQueueLength;

private:
    std::vector<AscendStreamPuller::VideoFrameInfo> videoFrameInfos;
    std::vector<std::shared_ptr<AscendStreamPuller::StreamPuller>> streamPullers;
    std::vector<std::shared_ptr<AscendVideoDecoder::VideoDecoder>> videoDecoders;

    std::shared_ptr<AscendImageResizer::ImageResizer> imageResizer;
    std::shared_ptr<AscendResnetDetector::ResnetDetector> resnetDetector;
    std::shared_ptr<AscendFrameSkippingSampling::FrameSkippingSampling> frameSkippingSampling;

    std::map<int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> decodeFrameQueueMap;
};

#endif //MULTICHANNELVIDEODETECTION_VIDEOGESTUREREASONER_H
