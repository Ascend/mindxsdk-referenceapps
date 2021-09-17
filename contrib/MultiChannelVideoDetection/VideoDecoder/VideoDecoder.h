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

#ifndef MULTICHANNELVIDEODETECTION_VIDEODECODER_H
#define MULTICHANNELVIDEODETECTION_VIDEODECODER_H

#include "BlockingQueue/BlockingQueue.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"

namespace AscendVideoDecoder {
struct DecoderInitParam {
    uint32_t deviceId;
    uint32_t channelId;
    // the width of video frame which need to decode
    uint32_t inputVideoWidth;
    // the height of video frame which need to decode
    uint32_t inputVideoHeight;
    // the format of video which need to decode
    MxBase::MxbaseStreamFormat inputVideoFormat;
    // the format of output image
    MxBase::MxbasePixelFormat outputImageFormat;
};

class VideoDecoder {
public:
    VideoDecoder() = default;
    ~VideoDecoder() = default;

    APP_ERROR Init(const DecoderInitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Decode(MxBase::MemoryData &streamData,
                     const std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> &decodeFrameQueue);

    uint32_t GetTotalDecodeFrameNum() const;

private:
    APP_ERROR InitDvppWrapper(const DecoderInitParam &initParam);
    static APP_ERROR VideoDecodeCallback(std::shared_ptr<void> buffer,
                                         MxBase::DvppDataInfo &inputDataInfo, void *userData);

private:
    // channel id
    uint32_t channelId = 0;
    // device id
    uint32_t deviceId = 0;

    // curr video frame id
    uint32_t frameId = 0;
    // video width
    uint32_t frameWidth = 0;
    // video height
    uint32_t frameHeight = 0;

    // video decoder (MX SDK)
    std::shared_ptr<MxBase::DvppWrapper> vDvppWrapper;
};
} // end AscendVideoDecoder
#endif // MULTICHANNELVIDEODETECTION_VIDEODECODER_H