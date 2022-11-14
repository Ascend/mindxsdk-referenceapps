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

#include "MediaCodecV2.h"

#include "MxBase/Log/Log.h"
#include <algorithm>
#include <map>
#include "MxBase/Maths/FastMath.h"
#include<condition_variable>
#include <signal.h> // signal functions

using namespace MxBase;
using namespace std;
namespace {
    const uint32_t MAX_WIDTH = 3840;
    const uint32_t MAX_HEIGHT = 2160;
    const uint32_t SRC_WIDTH = 1920;
    const uint32_t SRC_HEIGHT = 1080;
    const uint32_t RESIZE_WIDTH = 1280;
    const uint32_t RESIZE_HEIGHT = 720;
    const uint32_t MS_TIMEOUT = 39;
    const uint32_t KEY_FRAME_INTERVAL = 50;
    const uint32_t SRC_RATE = 25;
    const uint32_t RC_MODE = 0;
    const uint32_t MAX_BIT_RATE = 2080;
    const uint32_t IP_PROP = 50;
    const uint32_t DEVICE_ID = 0;
    const uint32_t CHANNEL_ID = 1;
    AVFormatContext *pFormatCtx = nullptr;
}

APP_ERROR MediaCodecv2::Init(std::string filePath, std::string savePath)
{
    openFilePath = filePath;
    saveFilePath = savePath;
    PullStream(filePath);
    // imageProcess init
    imageProcessorDptr = std::make_shared<MxBase::ImageProcessor>(DEVICE_ID);
    if (imageProcessorDptr == nullptr)
    {
        LogError << "imageProcessorDptr nullptr";
    }

    return APP_ERR_OK;
}

void MediaCodecv2::PullStream(std::string filePath)
{
    LogInfo << "start to PullStream";
    avformat_network_init();
    pFormatCtx = avformat_alloc_context();
    pFormatCtx = CreateFormatContext(filePath);
    av_dump_format(pFormatCtx, 0, filePath.c_str(), 0);
}

// ffmpeg 拉流
AVFormatContext* MediaCodecv2::CreateFormatContext(std::string filePath)
{
    LogInfo << "start to CreatFormatContext!";
    // creat message for stream pull
    AVFormatContext *formatContext = nullptr;
    AVDictionary *options = nullptr;

    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "stimeout", "3000000", 0);

    std::string str = filePath.substr(filePath.find_last_of('.') + 1);
    if (str != "h264" && str != "264") {
        LogError << "Couldn't decode " << str << " file";
        return nullptr;
    }

    LogInfo << "start to avformat_open_input!";
    int ret = avformat_open_input(&formatContext, filePath.c_str(), nullptr, &options);
    if (options != nullptr)
    {
        av_dict_free(&options);
    }
    if (ret != 0)
    {
        LogError << "Couldn`t open input stream " << filePath.c_str() << " ret=" << ret;
        return nullptr;
    }
    ret = avformat_find_stream_info(formatContext, nullptr);
    if (ret != 0)
    {
        LogError << "Couldn`t open input stream information";
        return nullptr;
    }

    return formatContext;
}

// 获取H264中的帧
void MediaCodecv2::GetFrame(AVPacket& pkt, FrameImage& frameImage, AVFormatContext* pFormatCtx)
{
    LogInfo << "start to GetFrame";
    av_init_packet(&pkt);
    int ret = av_read_frame(pFormatCtx, &pkt);
    if (ret != 0)
    {
        LogInfo << "[StreamPuller] channel Read frame failed, continue!";
        if (ret == AVERROR_EOF)
        {
            LogInfo << "[StreamPuller] channel StreamPuller is EOF, over!";
            stopFlag = true;
            return;
        }
    } else {
        if (pkt.size <= 0)
        {
            LogError << "Invalid pkt.size: " << pkt.size;
            return;
        }

        // send to the device
        auto hostDeleter = [](void *dataPtr) -> void { };
        MemoryData data(pkt.size, MemoryData::MEMORY_HOST);
        MemoryData src((void *)(pkt.data), pkt.size, MemoryData::MEMORY_HOST);
        APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(data, src);
        if (ret != APP_ERR_OK)
        {
            LogError << "MxbsMallocAndCopy failed!";
        }
        std::shared_ptr<uint8_t> imageData((uint8_t*)data.ptrData, hostDeleter);

        Image subImage(imageData, pkt.size);
        frameImage.image = subImage;

        LogInfo << "'channelId = " << frameImage.channelId << ", frameId = " << frameImage.frameId << " , dataSize = "
                << frameImage.image.GetDataSize();

        av_packet_unref(&pkt);
    }
    LogInfo << "end to GetFrame";
    return;
}

APP_ERROR MediaCodecv2::Resize(const MxBase::Image &decodedImage, MxBase::Image &resizeImage)
{
    std::shared_ptr<MxBase::ImageProcessor> imageProcessorDptr;
    imageProcessorDptr = std::make_shared<MxBase::ImageProcessor>(DEVICE_ID);
    // set size param
    Size resizeConfig(RESIZE_WIDTH, RESIZE_HEIGHT);
    APP_ERROR ret;
    ret = imageProcessorDptr->Resize(decodedImage, resizeConfig, resizeImage, Interpolation::HUAWEI_HIGH_ORDER_FILTER);
    if (ret != APP_ERR_OK)
    {
        LogError << "Resize failed, ret= " << ret;
        return ret;
    }
    LogInfo << "ReszieWidth = " << RESIZE_WIDTH << ", ResizeHight = " << RESIZE_HEIGHT;
    return APP_ERR_OK;
}

void MediaCodecv2::PullStreamThread()
{
    LogInfo << "start to pull stream thread.";
    AVPacket pkt;
    uint32_t frameId = 0;

    MxBase::DeviceContext device;
    device.devId = DEVICE_ID;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->SetDevice(device);
    if (ret != APP_ERR_OK) {
        LogError << "SetDevice failed";
        return;
    }

    while (!stopFlag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(MS_TIMEOUT));
        Image subImage;
        FrameImage frame;
        frame.channelId = 0;
        frame.frameId = frameId;
        frame.image = subImage;

        MediaCodecv2::GetFrame(pkt, frame, pFormatCtx);
        pullStreamQueue.Push(frame);
    }
    return;
}

void MediaCodecv2::DecodeThread()
{
    LogInfo << "start to video decode thread.";
    int32_t channelId = CHANNEL_ID;
    // 解码器参数
    VideoDecodeConfig vDconfig;
    VideoDecodeCallBack cPtr = MxBase::CallBackVdecode;
    vDconfig.width = SRC_WIDTH;
    vDconfig.height = SRC_HEIGHT;
    vDconfig.callbackFunc = cPtr;
    vDconfig.skipInterval = 0; // 跳帧控制

    std::shared_ptr<VideoDecoder> videoDecoder = std::make_shared<VideoDecoder>(vDconfig, DEVICE_ID, channelId);
    while (!stopFlag) {
        FrameImage temp;
        pullStreamQueue.Pop(temp);
        if (temp.image.GetDataSize() != APP_ERR_OK)
        {
            APP_ERROR ret = videoDecoder->Decode(temp.image.GetData(), temp.image.GetDataSize(), frameId, &decodeQueue);
            if (ret != 0)
            {
                LogError << "videoDecoder Decode failed. ret is: " << ret;
            }
        } else {
            break;
        }

        frameId += 1;
    }
    return;
}

// 视频解码回调
APP_ERROR MxBase::CallBackVdecode(Image& decodedImage, uint32_t channelId, uint32_t frameId, void* userData)
{
    FrameImage frameImage;
    frameImage.image = decodedImage;
    frameImage.channelId = channelId;

    BlockingQueue<FrameImage> *p_decodeQueue = (BlockingQueue<FrameImage> *)userData;
    p_decodeQueue->Push(frameImage);

    return APP_ERR_OK;
};

// resize video frame
void MediaCodecv2::ResizeThread()
{
    LogInfo << "start to resize resize thread.";
    while (!stopFlag) {
        FrameImage decodeTemp;
        decodeQueue.Pop(decodeTemp);

        MxBase::Image resizeImage;
        Resize(decodeTemp.image, resizeImage);

        FrameImage frameImage;
        frameImage.image = resizeImage;
        frameImage.channelId = channelId;

        resizeQueue.Push(frameImage);
    }
    return;
}

void MediaCodecv2::EncodeThread()
{
    LogInfo << "start to video encode thread.";
    VideoEncodeConfig vEConfig;
    VideoEncodeCallBack cEPtr = MxBase::CallBackVencode;
    vEConfig.callbackFunc = cEPtr;
    vEConfig.width = RESIZE_WIDTH;
    vEConfig.height = RESIZE_HEIGHT;
    // 用户可自定义编码参数
    vEConfig.keyFrameInterval = KEY_FRAME_INTERVAL;
    vEConfig.srcRate = SRC_RATE;
    vEConfig.rcMode = RC_MODE;
    vEConfig.maxBitRate = MAX_BIT_RATE;
    vEConfig.ipProp = IP_PROP;
    std::shared_ptr<VideoEncoder> videoEncoder = std::make_shared<VideoEncoder>(vEConfig, DEVICE_ID);
    int frameCount = 0;
    while (!stopFlag) {
        FrameImage resizeTemp;
        resizeQueue.Pop(resizeTemp);

        videoEncoder->Encode(resizeTemp.image, resizeTemp.frameId, &encodeQueue);
        frameCount += 1;
    }
    return;
}

// 视频编码回调
APP_ERROR MxBase::CallBackVencode(std::shared_ptr<uint8_t>& outDataPtr, uint32_t& outDataSize,  uint32_t& channelId,
                                  uint32_t& frameId, void* userData)
{
    Image image(outDataPtr, outDataSize, -1, Size(MAX_WIDTH, MAX_HEIGHT));
    FrameImage frameImage;
    frameImage.image = image;
    frameImage.channelId = channelId;
    frameImage.frameId = frameId;

    LogInfo << "frameId(" << frameImage.frameId << ") encoded successfully.";

    bool bIsIDR = (outDataSize > 1);
    if (frameImage.frameId)
    {
        if (!bIsIDR)
        {
            LogError << "Not bIsIDR!";
            return APP_ERR_OK;
        }
    }

    BlockingQueue<FrameImage> *p_encodeQueue = (BlockingQueue<FrameImage> *)userData;
    p_encodeQueue->Push(frameImage);

    return APP_ERR_OK;
};

// record frame number per second
void MediaCodecv2::CalFps()
{
    while (!stopFlag) {
        sleep(1);
        LogInfo << "video encode frame rate for per second: " << finishCount - lastCount << " fps.";
        lastCount = finishCount;
    }
}

// write result
void MediaCodecv2::WriteThread()
{
    FILE *fp = fopen(saveFilePath.c_str(), "wb");

    while (!stopFlag) {
        FrameImage encodeTemp;
        encodeQueue.Pop(encodeTemp);

        std::shared_ptr<uint8_t> data_sp = encodeTemp.image.GetData();
        void *data_p = data_sp.get();
        if (fp == nullptr)
        {
            LogError << "Failed to open file.";
            return;
        }

        if (fwrite(data_p, encodeTemp.image.GetDataSize(), 1, fp) != 1)
        {
            LogInfo << "write frame to file fail";
        }

        finishCount++;
    }
    return;
}

// process thread
APP_ERROR MediaCodecv2::Process(std::string filePath, std::string outPath)
{
    auto startTime = std::chrono::high_resolution_clock::now();

    std::thread threadPullStream(&MediaCodecv2::PullStreamThread, this);

    std::thread threadDecode(&MediaCodecv2::DecodeThread, this);

    std::thread threadResize(&MediaCodecv2::ResizeThread, this);

    std::thread threadEncode(&MediaCodecv2::EncodeThread, this);

    std::thread threadWrite(&MediaCodecv2::WriteThread, this);

    std::thread threadCalFps(&MediaCodecv2::CalFps, this);

    threadPullStream.join();
    threadDecode.join();
    threadResize.join();
    threadEncode.join();
    threadWrite.join();
    threadCalFps.join();

    auto endTime = std::chrono::high_resolution_clock::now();
    double costS = std::chrono::duration<double>(endTime - startTime).count();
    LogInfo << "total process time: " << costS << "s.";
    double fps = finishCount / costS;
    LogInfo << "Total decode frame rate: " << fps << " fps.";
    return APP_ERR_OK;
}

// stop process
void MediaCodecv2::stopProcess()
{
    stopFlag = true;
}
