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

#include "StreamPuller.h"
#include "MxBase/Log/Log.h"
#include <thread>

namespace AscendStreamPuller {
namespace {
    const uint32_t LOW_THRESHOLD = 0;
    const uint32_t MAX_THRESHOLD = 4096;
}

APP_ERROR StreamPuller::Init(const std::string &rtspUrl, uint32_t deviceId)
{
    LogDebug << "StreamPuller" << ": StreamPuller init start.";

    this->deviceId = deviceId;
    this->streamName = rtspUrl;
    this->frameInfo.source = rtspUrl;

    stopFlag = false;

    APP_ERROR ret = TryToStartStream();
    if (ret != APP_ERR_OK) {
        LogError << "start stream failed";
        return ret;
    }

    LogDebug << "StreamPuller" << ": StreamPuller init success.";
    return APP_ERR_OK;
}

APP_ERROR StreamPuller::DeInit()
{
    LogDebug << "StreamPuller" << ": StreamPuller deinit start.";
    AVFormatContext* pAvFormatContext = formatContext.get();
    avformat_close_input(&pAvFormatContext);

    stopFlag = true;
    formatContext = nullptr;
    LogDebug << "StreamPuller" << ": StreamPuller deinit success.";
    return APP_ERR_OK;
}

APP_ERROR StreamPuller::Process()
{
    PullStreamDataLoop();
    return APP_ERR_OK;
}

MxBase::MemoryData StreamPuller::GetNextFrame()
{
    AVPacket packet;

    av_init_packet(&packet);
    while (true) {
        if (stopFlag || formatContext == nullptr) {
            LogDebug << "StreamPuller stopped or deinit, pull video stream exit";
            break;
        }

        APP_ERROR ret = av_read_frame(formatContext.get(), &packet);
        if (ret != APP_ERR_OK) {
            if (ret == AVERROR_EOF) {
                LogDebug << "StreamPuller channel StreamPuller is EOF, over!";
                stopFlag = true;
                break;
            }

            LogError << "StreamPuller channel Read frame failed, continue!";
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            continue;
        } else if (packet.stream_index != frameInfo.videoStream) {
            LogDebug << "packet is not video stream. continue";
            av_packet_unref(&packet);
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            continue;
        }

        if (packet.size <= 0) {
            LogError << "Invalid packet.size: " << packet.size;
            continue;
        }

        // deep copy packet data
        auto packetData = new uint8_t[(size_t) packet.size + 1];
        memcpy(packetData, packet.data, (size_t) packet.size);

        // put video frame into queue
        MxBase::MemoryData streamData((void *) packetData, (size_t) packet.size,
                                      MxBase::MemoryData::MEMORY_HOST_NEW, deviceId);

        av_packet_unref(&packet);
        return streamData;
    }

    av_packet_unref(&packet);

    return {nullptr, 0, MxBase::MemoryData::MEMORY_HOST_NEW, deviceId};
}

VideoFrameInfo StreamPuller::GetFrameInfo()
{
    return frameInfo;
}

/// ========== Private Method ========== ///
APP_ERROR StreamPuller::TryToStartStream()
{
    uint32_t failureNum = 0;
    while (failureNum < maxReTryOpenStream) {
        APP_ERROR ret = StartStream();
        if (ret == APP_ERR_OK) {
            LogDebug << "StreamPuller start stream success.";
            ret = GetStreamInfo();
            if (ret != APP_ERR_OK) {
                LogError << "StreamPuller get stream info error";
                return ret;
            }
            return APP_ERR_OK;
        }
        LogError << "StreamPuller start stream failed, retry: " << ++failureNum;
    }

    stopFlag = true;
    return APP_ERR_COMM_INIT_FAIL;
}

APP_ERROR StreamPuller::StartStream()
{
    // init network
    avformat_network_init();

    // malloc avformat context
    AVFormatContext* pAvformatContext = avformat_alloc_context();
    formatContext = std::shared_ptr<AVFormatContext>(pAvformatContext);
    if (formatContext == nullptr) {
        LogError << "formatContext is null.";
        return APP_ERR_COMM_INVALID_POINTER;
    }

    APP_ERROR ret = CreateFormatContext();
    if (ret != APP_ERR_OK) {
        LogError << "Couldn't create format context" << " ret = " << ret;
        return ret;
    }

    // for debug dump
    av_dump_format(formatContext.get(), 0, streamName.c_str(), 0);
    return APP_ERR_OK;
}

APP_ERROR StreamPuller::CreateFormatContext()
{
    // create message for stream pull
    AVDictionary *options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "stimeout", "3000000", 0);

    AVFormatContext* pAvformatContext = formatContext.get();
    APP_ERROR ret = avformat_open_input(&pAvformatContext, streamName.c_str(), nullptr, &options);
    if (options != nullptr) {
        av_dict_free(&options);
    }

    if(ret != APP_ERR_OK) {
        LogError << "Couldn't open input stream " << streamName.c_str() <<  " ret = " << ret;
        return APP_ERR_STREAM_NOT_EXIST;
    }

    ret = avformat_find_stream_info(formatContext.get(), nullptr);
    if(ret != APP_ERR_OK) {
        LogError << "Couldn't find stream information" << " ret = " << ret;
        return APP_ERR_STREAM_NOT_EXIST;
    }

    return APP_ERR_OK;
}

APP_ERROR StreamPuller::GetStreamInfo()
{
    frameInfo.videoStream = -1;

    if (formatContext != nullptr) {
        for (uint32_t i = 0; i < formatContext->nb_streams; i++) {
            AVStream* inStream = formatContext->streams[i];
            if (inStream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                frameInfo.videoStream = i;
                frameInfo.width = inStream->codecpar->width;
                frameInfo.height = inStream->codecpar->height;
                break;
            }
        }
        if (frameInfo.videoStream == -1) {
            LogError << "Didn't find a video stream!";
            return APP_ERR_COMM_FAILURE;
        }

        AVCodecID codecId = formatContext->streams[frameInfo.videoStream]->codecpar->codec_id;
        if (codecId == AV_CODEC_ID_H264) {
            frameInfo.format = MxBase::MXBASE_STREAM_FORMAT_H264_MAIN_LEVEL;
        } else if (codecId == AV_CODEC_ID_H265){
            frameInfo.format = MxBase::MXBASE_STREAM_FORMAT_H265_MAIN_LEVEL;
        } else {
            LogError << "\033[0;31mError unsupported format \033[0m" << codecId;
            return APP_ERR_COMM_FAILURE;
        }
    }

    // check video frame size
    if (frameInfo.width < LOW_THRESHOLD || frameInfo.height < LOW_THRESHOLD ||
        frameInfo.width > MAX_THRESHOLD || frameInfo.height > MAX_THRESHOLD) {
        LogError << "Size of frame is not supported in DVPP Video Decode!";
        return APP_ERR_COMM_FAILURE;
    }

    return APP_ERR_OK;
}

void StreamPuller::PullStreamDataLoop()
{
    while (true) {
        if (stopFlag || formatContext == nullptr) {
            LogDebug << "StreamPuller stopped or deinit, pull video stream exit";
            break;
        }
        MxBase::MemoryData videoFrame = GetNextFrame();

        if (videoFrame.size == 0 || videoFrame.ptrData == nullptr) {
            LogDebug << "empty video frame, not need! continue!";
            continue;
        }

        // todo send stream data to Device

    }
}
} // end AscendStreamPuller