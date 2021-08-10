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

#include <thread>
#include "MxBase/ErrorCode/ErrorCodes.h"
#include "MxBase/Log/Log.h"
#include "opencv2/opencv.hpp"
#include "VideoProcess.h"

namespace {
    static AVFormatContext *formatContext = nullptr; // 视频流信息
    const uint32_t VIDEO_WIDTH = {视频宽度};
    const uint32_t VIDEO_HEIGHT = {视频高度};
    const uint32_t MAX_QUEUE_LENGHT = 1000;
    const uint32_t QUEUE_POP_WAIT_TIME = 10;
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
}

APP_ERROR VideoProcess::StreamInit(const std::string &rtspUrl)
{
    avformat_network_init();

    AVDictionary *options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "stimeout", "3000000", 0);
    // ffmpeg打开流媒体-视频流
    APP_ERROR ret = avformat_open_input(&formatContext, rtspUrl.c_str(), nullptr, &options);
    if (options != nullptr) {
        av_dict_free(&options);
    }
    if(ret != APP_ERR_OK){
        LogError << "Couldn't open input stream " << rtspUrl.c_str() <<  " ret = " << ret;
        return APP_ERR_STREAM_NOT_EXIST;
    }
    // 获取视频的相关信息
    ret = avformat_find_stream_info(formatContext, nullptr);
    if(ret != APP_ERR_OK){
        LogError << "Couldn't find stream information";
        return APP_ERR_STREAM_NOT_EXIST;
    }
    // 打印视频信息
    av_dump_format(formatContext, 0, rtspUrl.c_str(), 0);
    return APP_ERR_OK;
}

APP_ERROR VideoProcess::StreamDeInit()
{
    avformat_close_input(&formatContext);
    return APP_ERR_OK;
}

// 每进行一次视频帧解码会调用一次该函数，将解码后的帧信息存入对列中
APP_ERROR VideoProcess::VideoDecodeCallback(std::shared_ptr<void> buffer, MxBase::DvppDataInfo &inputDataInfo, 
                                            void *userData)
{
    auto deleter = [] (MxBase::MemoryData *mempryData) {
        if (mempryData == nullptr) {
            LogError << "MxbsFree failed";
            return;
        }
        APP_ERROR ret = MxBase::MemoryHelper::MxbsFree(*mempryData);
        delete mempryData;
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << " MxbsFree failed";
            return;
        }
        LogInfo << "MxbsFree successfully";
    };
    // 解码后的视频信息
    auto output = std::shared_ptr<MxBase::MemoryData>(new MxBase::MemoryData(buffer.get(),
                     (size_t)inputDataInfo.dataSize, MxBase::MemoryData::MEMORY_DVPP, inputDataInfo.frameId), deleter);

    if (userData == nullptr) {
        LogError << "userData is nullptr";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    auto *queue = (BlockingQueue<std::shared_ptr<void>>*)userData;
    queue->Push(output);
    return APP_ERR_OK;
}

APP_ERROR VideoProcess::VideoDecodeInit()
{
    MxBase::VdecConfig vdecConfig;
    // 将解码函数的输入格式设为H264
    vdecConfig.inputVideoFormat = MxBase::MXBASE_STREAM_FORMAT_H264_MAIN_LEVEL;
    // 将解码函数的输出格式设为YUV420
    vdecConfig.outputImageFormat = MxBase::MXBASE_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    vdecConfig.deviceId = DEVICE_ID;
    vdecConfig.channelId = CHANNEL_ID;
    vdecConfig.callbackFunc = VideoDecodeCallback;
    vdecConfig.outMode = 1;

    vDvppWrapper = std::make_shared<MxBase::DvppWrapper>();
    if (vDvppWrapper == nullptr) {
        LogError << "Failed to create dvppWrapper";
        return APP_ERR_COMM_INIT_FAIL;
    }
    APP_ERROR ret = vDvppWrapper->InitVdec(vdecConfig);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to initialize dvppWrapper";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR VideoProcess::VideoDecodeDeInit()
{
    APP_ERROR ret = vDvppWrapper->DeInitVdec();
    if (ret != APP_ERR_OK) {
        LogError << "Failed to deinitialize dvppWrapper";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR VideoProcess::VideoDecode(MxBase::MemoryData &streamData, const uint32_t &height, 
                                    const uint32_t &width, void *userData)
{
    static uint32_t frameId = 0;
    // 将帧数据从Host侧移到Device侧
    MxBase::MemoryData dvppMemory((size_t)streamData.size,
                                  MxBase::MemoryData::MEMORY_DVPP, DEVICE_ID);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(dvppMemory, streamData);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to MxbsMallocAndCopy";
        return ret;
    }
    // 构建DvppDataInfo结构体以便解码
    MxBase::DvppDataInfo inputDataInfo;
    inputDataInfo.dataSize = dvppMemory.size;
    inputDataInfo.data = (uint8_t *)dvppMemory.ptrData;
    inputDataInfo.height = VIDEO_HEIGHT;
    inputDataInfo.width = VIDEO_WIDTH;
    inputDataInfo.channelId = CHANNEL_ID;
    inputDataInfo.frameId = frameId;
    ret = vDvppWrapper->DvppVdec(inputDataInfo, userData);

    if (ret != APP_ERR_OK) {
        LogError << "DvppVdec Failed";
        MxBase::MemoryHelper::MxbsFree(dvppMemory);
        return ret;
    }
    frameId++;
    return APP_ERR_OK;
}

// 获取视频帧
void VideoProcess::GetFrames(std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>  blockingQueue, 
                            std::shared_ptr<VideoProcess> videoProcess)
{
    MxBase::DeviceContext device;
    device.devId = DEVICE_ID;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->SetDevice(device);
    if (ret != APP_ERR_OK) {
        LogError << "SetDevice failed";
        return;
    }

    AVPacket pkt;
    while(!stopFlag){
        av_init_packet(&pkt);
        // 读取视频帧
        APP_ERROR ret = av_read_frame(formatContext, &pkt);
        if(ret != APP_ERR_OK){
            LogError << "Read frame failed, continue";
            if(ret == AVERROR_EOF){
                LogError << "StreamPuller is EOF, over!";
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // 原始帧数据被存储在Host侧
        MxBase::MemoryData streamData((void *)pkt.data, (size_t)pkt.size,
                                      MxBase::MemoryData::MEMORY_HOST_NEW, DEVICE_ID);
        ret = videoProcess->VideoDecode(streamData, VIDEO_HEIGHT, VIDEO_WIDTH, (void*)blockingQueue.get());
        if (ret != APP_ERR_OK) {
            LogError << "VideoDecode failed";
            return;
        }
        av_packet_unref(&pkt);
    }
    av_packet_unref(&pkt);
}

APP_ERROR VideoProcess::SaveResult(std::shared_ptr<MxBase::MemoryData> resultInfo, const uint32_t frameId,
                     const std::vector<std::vector<MxBase::ObjectInfo>> objInfos)
{
    // 将推理结果从Device侧移到Host侧
    MxBase::MemoryData memoryDst(resultInfo->size,MxBase::MemoryData::MEMORY_HOST_NEW);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst, *resultInfo);
    if(ret != APP_ERR_OK){
        LogError << "Fail to malloc and copy host memory.";
        return ret;
    }
    // 初始化OpenCV图像信息矩阵
    cv::Mat imgYuv = cv::Mat(VIDEO_HEIGHT* YUV_BYTE_NU / YUV_BYTE_DE, VIDEO_WIDTH, CV_8UC1, memoryDst.ptrData);
    cv::Mat imgBgr = cv::Mat(VIDEO_HEIGHT, VIDEO_WIDTH, CV_8UC3);
    // 颜色空间转换
    cv::cvtColor(imgYuv, imgBgr, cv::COLOR_YUV2BGR_NV12);

    std::vector<MxBase::ObjectInfo> info;
    for (uint32_t i = 0; i < objInfos.size(); i++) {
        float maxConfidence = 0;
        uint32_t index;
        for (uint32_t j = 0; j < objInfos[i].size(); j++) {
            if (objInfos[i][j].confidence > maxConfidence) {
                maxConfidence = objInfos[i][j].confidence;
                index = j;
            }
        }
        info.push_back(objInfos[i][index]);
        // 打印置信度最大推理结果
        LogInfo << "id: " << info[i].classId << "; lable: " << info[i].className
              << "; confidence: " << info[i].confidence
              << "; box: [ (" << info[i].x0 << "," << info[i].y0 << ") "
              << "(" << info[i].x1 << "," << info[i].y1 << ") ]";

        const cv::Scalar green = cv::Scalar(0, 255, 0);
        const uint32_t thickness = 4;
        const uint32_t xOffset = 10;
        const uint32_t yOffset = 10;
        const uint32_t lineType = 8;
        const float fontScale = 1.0;

        // 在图像上绘制文字
        cv::putText(imgBgr, info[i].className, cv::Point(info[i].x0 + xOffset, info[i].y0 + yOffset),
                        cv::FONT_HERSHEY_SIMPLEX, fontScale, green, thickness, lineType);
        // 绘制矩形
        cv::rectangle(imgBgr,cv::Rect(info[i].x0, info[i].y0,
                                      info[i].x1 - info[i].x0, info[i].y1 - info[i].y0),
                      green, thickness);
        // 把Mat类型的图像矩阵保存为图像到指定位置。
        std::string fileName = "./result/result" + std::to_string(frameId+1) + ".jpg";
        cv::imwrite(fileName, imgBgr);
    }

    ret = MxBase::MemoryHelper::MxbsFree(memoryDst);
    if(ret != APP_ERR_OK){
        LogError << "Fail to MxbsFree memory.";
        return ret;
    }
    return APP_ERR_OK;
}

void VideoProcess::GetResults(std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> blockingQueue, 
                              std::shared_ptr<Yolov3Detection> yolov3Detection,
                              std::shared_ptr<VideoProcess> videoProcess)
{
    uint32_t frameId = 0;
    MxBase::DeviceContext device;
    device.devId = DEVICE_ID;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->SetDevice(device);
    if (ret != APP_ERR_OK) {
        LogError << "SetDevice failed";
        return;
    }
    while (!stopFlag) {
        std::shared_ptr<void> data = nullptr;
        // 从队列中去出解码后的帧数据
        APP_ERROR ret = blockingQueue->Pop(data, QUEUE_POP_WAIT_TIME);
        if (ret != APP_ERR_OK) {
            LogError << "Pop failed";
            return;
        }
        LogInfo << "get result";

        MxBase::TensorBase resizeFrame;
        auto result = std::make_shared<MxBase::MemoryData>();
        result = std::static_pointer_cast<MxBase::MemoryData>(data);

        // 图像缩放
        ret = yolov3Detection->ResizeFrame(result, VIDEO_HEIGHT, VIDEO_WIDTH,resizeFrame);
        if (ret != APP_ERR_OK) {
            LogError << "Resize failed";
            return;
        }

        std::vector<MxBase::TensorBase> inputs = {};
        std::vector<MxBase::TensorBase> outputs = {};
        inputs.push_back(resizeFrame);
        // 推理
        ret = yolov3Detection->Inference(inputs, outputs);
        if (ret != APP_ERR_OK) {
            LogError << "Inference failed, ret=" << ret << ".";
            return;
        }

        std::vector<std::vector<MxBase::ObjectInfo>> objInfos;
        // 后处理
        ret = yolov3Detection->PostProcess(outputs, VIDEO_HEIGHT, VIDEO_WIDTH, objInfos);
        if (ret != APP_ERR_OK) {
            LogError << "PostProcess failed, ret=" << ret << ".";
            return;
        }

        // 结果可视化
        ret = videoProcess->SaveResult(result, frameId, objInfos);
        if (ret != APP_ERR_OK) {
            LogError << "Save result failed, ret=" << ret << ".";
            return;
        }
        frameId++;

    }
}