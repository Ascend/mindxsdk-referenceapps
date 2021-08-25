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
    const uint32_t LOW_THRESHOLD = 128;
    const uint32_t MAX_THRESHOLD = 4096;
}

/**
 * Init StreamPuller
 * @param rtspUrl const reference to rtsp stream video address
 * @param maxTryOpenStreamTimes max times of retrying to open video Stream
 * @param deviceId device id
 * @return status code of whether initialization is successful
 */
APP_ERROR StreamPuller::Init(const std::string &rtspUrl, uint32_t maxTryOpenStreamTimes, uint32_t deviceId)
{
    LogInfo << "StreamPuller init start.";

    this->deviceId = deviceId;
    this->streamName = rtspUrl;
    this->maxReTryOpenStream = maxTryOpenStreamTimes;
    this->frameInfo.source = rtspUrl;

    stopFlag = false;

    APP_ERROR ret = TryToStartStream();
    if (ret != APP_ERR_OK) {
        LogError << "start stream failed.";
        return ret;
    }

    LogInfo << "StreamPuller init success.";
    return APP_ERR_OK;
}

/**
 * De-init StreamPuller
 * @return status code of whether de-initialization is successful
 */
APP_ERROR StreamPuller::DeInit()
{
    LogInfo << "StreamPuller deinit start.";
    AVFormatContext* pAvFormatContext = formatContext.get();
    avformat_close_input(&pAvFormatContext);

    stopFlag = true;
    formatContext = nullptr;
    LogInfo << "StreamPuller deinit success.";
    return APP_ERR_OK;
}

/**
 * Get the next frame data of video stream
 * @return the memory data of next frame
 */
MxBase::MemoryData StreamPuller::GetNextFrame()
{
    AVPacket packet;

    av_init_packet(&packet);
    while (true) {
        if (stopFlag || formatContext == nullptr) {
            LogInfo << "StreamPuller stopped or deinit, pull video stream exit.";
            break;
        }

        APP_ERROR ret = av_read_frame(formatContext.get(), &packet);
        if (ret != APP_ERR_OK) {
            if (ret == AVERROR_EOF) {
                LogInfo << "StreamPuller channel StreamPuller is EOF, over!";
                stopFlag = true;
                break;
            }

            LogError << "StreamPuller channel Read frame failed, continue!";
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        } else if (packet.stream_index != frameInfo.videoStream) {
            LogDebug << "packet is not video stream. continue.";
            av_packet_unref(&packet);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        if (packet.size <= 0) {
            LogError << "Invalid packet.size: " << packet.size << ".";
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

/**
 * Get video stream frame info
 * @return {@link VideoFrameInfo}
 */
VideoFrameInfo StreamPuller::GetFrameInfo()
{
    return frameInfo;
}

/// ========== Private Method ========== ///
/**
 * Try to start the video stream
 * >> first step: alloc context
 * >> second step: get video stream frame info
 * >> and check video format and size
 * @return status code of whether start stream is successful
 */
APP_ERROR StreamPuller::TryToStartStream()
{
    uint32_t failureNum = 0;
    while (failureNum < maxReTryOpenStream) {
        APP_ERROR ret = StartStream();
        if (ret == APP_ERR_OK) {
            LogDebug << "StreamPuller start stream success.";
            ret = GetStreamInfo();
            if (ret != APP_ERR_OK) {
                LogError << "StreamPuller get stream info error.";
                return ret;
            }
            return APP_ERR_OK;
        }
        LogError << "StreamPuller start stream failed, retry: " << ++failureNum << ".";
    }

    stopFlag = true;
    return APP_ERR_COMM_INIT_FAIL;
}

/**
 * Start video stream: alloc context and print debug message
 * @return status code of whether start stream is successful
 */
APP_ERROR StreamPuller::StartStream()
{
    // init network
    avformat_network_init();

    // specify an empty deleter to avoid double free
    auto deleter = [] (AVFormatContext* avFormatContext) {

    };
    // malloc avformat context
    AVFormatContext* pAvformatContext = avformat_alloc_context();
    formatContext = std::shared_ptr<AVFormatContext>(pAvformatContext, deleter);
    if (formatContext == nullptr) {
        LogError << "formatContext is null.";
        return APP_ERR_COMM_INVALID_POINTER;
    }

    APP_ERROR ret = CreateFormatContext();
    if (ret != APP_ERR_OK) {
        LogError << "Couldn't create format context" << " ret = " << ret << ".";
        return ret;
    }

    // for debug dump
    av_dump_format(formatContext.get(), 0, streamName.c_str(), 0);
    return APP_ERR_OK;
}

/**
 * Open video input and find video stream info by context
 * @return status code of whether open video input and find stream info are successful
 */
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
        LogError << "Couldn't open input stream " << streamName.c_str() <<  " ret = " << ret << ".";
        return APP_ERR_STREAM_NOT_EXIST;
    }

    ret = avformat_find_stream_info(formatContext.get(), nullptr);
    if(ret != APP_ERR_OK) {
        LogError << "Couldn't find stream information" << " ret = " << ret << ".";
        return APP_ERR_STREAM_NOT_EXIST;
    }

    return APP_ERR_OK;
}

/**
 * Get video stream index, frame format and check frame size
 * @return status code of whether find and check video frame format and size are successful
 */
APP_ERROR StreamPuller::GetStreamInfo()
{
    frameInfo.videoStream = -1;

    if (formatContext != nullptr) {
        for (uint32_t i = 0; i < formatContext->nb_streams; i++) {
            AVStream* inStream = formatContext->streams[i];
            // find video stream index
            if (inStream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                frameInfo.videoStream = (int32_t) i;
                frameInfo.width = inStream->codecpar->width;
                frameInfo.height = inStream->codecpar->height;
                break;
            }
        }
        if (frameInfo.videoStream == -1) {
            LogError << "Didn't find a video stream!";
            return APP_ERR_COMM_FAILURE;
        }

        // check video format
        AVCodecID codecId = formatContext->streams[frameInfo.videoStream]->codecpar->codec_id;
        if (codecId == AV_CODEC_ID_H264) {
            frameInfo.format = MxBase::MXBASE_STREAM_FORMAT_H264_MAIN_LEVEL;
        } else if (codecId == AV_CODEC_ID_H265){
            frameInfo.format = MxBase::MXBASE_STREAM_FORMAT_H265_MAIN_LEVEL;
        } else {
            LogError << "\033[0;31mError unsupported format \033[0m" << codecId << ".";
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

} // end AscendStreamPuller