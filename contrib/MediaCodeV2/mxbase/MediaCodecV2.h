/*
 * Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

#ifndef MXBASE_MEDIACODECV2
#define MXBASE_MEDIACODECV2
#include <iostream>
#include <map>
#include <fstream>
#include "unistd.h"
#include <memory>
#include <queue>
#include <thread>
#include "boost/filesystem.hpp"

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"

#include "MxBase/E2eInfer/ImageProcessor/ImageProcessor.h"
#include "MxBase/E2eInfer/VideoDecoder/VideoDecoder.h"
#include "MxBase/E2eInfer/VideoEncoder/VideoEncoder.h"
#include "MxBase/E2eInfer/DataType.h"
#include "MxBase/E2eInfer/Image/Image.h"

#include "MxBase/MxBase.h"
#include "MxBase/Log/Log.h"

#include "BlockingQueue.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

namespace MxBase {
    APP_ERROR CallBackVdecode(MxBase::Image& decodedImage, uint32_t channelId, uint32_t frameId, void* userData);
    APP_ERROR CallBackVencode(std::shared_ptr<uint8_t>& outDataPtr,
                              uint32_t& outDataSize,  uint32_t& channelId, uint32_t& frameId, void* userData);
}

class FrameImage {
public:
    MxBase::Image image;    // video Image Class
    uint32_t frameId;
    uint32_t deviceId;
    uint32_t channelId;
    FrameImage& operator=(FrameImage &cls) {
        this->image = cls.image;
        this->frameId = cls.frameId;
        this->deviceId = cls.deviceId;
        this->channelId = cls.channelId;

        return *this;
    }
};

class MediaCodecv2
{
public:
    APP_ERROR Init(std::string filePath, std::string savePath);
    APP_ERROR Resize(const MxBase::Image& decodedImage, MxBase::Image &resizeImage);
    APP_ERROR Process(std::string filePath, std::string outPath);
    void PullStream(std::string filePath);
    AVFormatContext* CreateFormatContext(std::string filePath);
    static void GetFrame(AVPacket& pkt, FrameImage& frameImage, AVFormatContext* pFormatCtx);

    void PullStreamThread();
    void DecodeThread();
    void ResizeThread();
    void EncodeThread();
    void WriteThread();
    void CalFps();
    void stopProcess();

public:
    BlockingQueue<FrameImage> pullStreamQueue;
    BlockingQueue<FrameImage> decodeQueue;
    BlockingQueue<FrameImage> resizeQueue;
    BlockingQueue<FrameImage> encodeQueue;

    unsigned int finishCount = 0;
    unsigned int lastCount = 0;

protected:
    uint32_t deviceId = 0;
    uint32_t frameId = 0;
    uint32_t channelId = 0;

    static bool stopFlag;

    std::string openFilePath;
    std::string saveFilePath;

    std::shared_ptr<MxBase::ImageProcessor> imageProcessorDptr;
};
#endif /* MXBASE_MEDIACODECV2 */
