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

#ifndef VIDEOGESTUREREASONER_STREAMPULLER_H
#define VIDEOGESTUREREASONER_STREAMPULLER_H

#include <MxBase/DvppWrapper/DvppWrapper.h>
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "../BlockingQueue/BlockingQueue.h"

extern "C" {
#include "libavformat/avformat.h"
}

namespace AscendStreamPuller {
struct VideoFrameInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    // video stream channel id
    int32_t videoStream = 0;
    // video format
    MxBase::MxbaseStreamFormat format;
    // video source
    std::string source;
};

class StreamPuller {
public:
    StreamPuller() = default;
    ~StreamPuller() = default;
    APP_ERROR Init(const std::string &rtspUrl, uint32_t deviceId);
    APP_ERROR DeInit();
    APP_ERROR Process();
    MxBase::MemoryData GetNextFrame();

    VideoFrameInfo GetFrameInfo() const;

public:
    // running flag
    bool stopFlag = true;

private:
    APP_ERROR TryToStartStream();
    APP_ERROR StartStream();
    APP_ERROR CreateFormatContext();
    APP_ERROR GetStreamInfo();
    void PullStreamDataLoop();

private:
    // rtsp stream source
    std::string streamName = {};
    // device id
    uint32_t deviceId;
    // max retry times
    uint32_t maxReTryOpenStream = 1;
    // video frame property
    VideoFrameInfo frameInfo = {};
    // ffmpeg class member
    std::shared_ptr<AVFormatContext> formatContext;
};
} // end AscendStreamPuller

#endif // VIDEOGESTUREREASONER_STREAMPULLER_H