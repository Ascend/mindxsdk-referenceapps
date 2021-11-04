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
#include <iostream>
#include <sstream>
#include <queue>
#include <list>
#include "MxBase/ErrorCode/ErrorCodes.h"
#include "MxBase/Log/Log.h"
#include "opencv2/opencv.hpp"
#include "VideoProcess.h"
extern "C" {
    #include "libavformat/avformat.h"
    #include "libavcodec/avcodec.h"
    #include "libavutil/avutil.h"
    #include "libswscale/swscale.h"
}

namespace {
    static std::vector<std::queue<center>> pts(10000); // 保存每个车辆轨迹的最新的20个bbox的中心点
    static std::vector<center> line = {center{0,100}, center{1280, 100}}; // 计数所用的线段
    static int counter = 0;
    static int counter_down = 0;
    static int counter_up = 0;
    static AVFormatContext *formatContext = nullptr; // 视频流信息
    static uint32_t cnt = 0;
    const uint32_t VIDEO_WIDTH = 1280;
    const uint32_t VIDEO_HEIGHT = 720;
    const uint32_t MAX_QUEUE_LENGHT = 1000;
    const uint32_t QUEUE_POP_WAIT_TIME = 10;
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
}

bool ccw(center A, center B, center C){
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
}

// 计算两根线是否相交
bool intersect(center A, center B, center C, center D){
    if((ccw(A, C, D) != ccw(B, C, D)) && (ccw(A, B, C) != ccw(A, B, D))){
        return true;
    }
    else{
        return false;
    }
}
// 生成随机颜色
VideoProcess::VideoProcess() {
    for(uint32_t i = 0; i < 200; i++){
        for(uint32_t j = 0; j<3; j++){
            color_num[i][j] = rand() % 256;
        }
    }
}

APP_ERROR VideoProcess::StreamInit(const std::string &rtspUrl)
{
    LogInfo<<"StreamInit";
    avformat_network_init();

    AVDictionary *options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "stimeout", "3000000", 0);
    // ffmpeg打开流媒体-视频流
    APP_ERROR ret = avformat_open_input(&formatContext, rtspUrl.c_str(), nullptr, &options);
    LogInfo<<formatContext->packet_size;
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

// 每进行一次视频帧解码会调用一次该函数，将解码后的帧信息存入队列中
APP_ERROR VideoProcess::VideoDecodeCallback(std::shared_ptr<void> buffer, MxBase::DvppDataInfo &inputDataInfo,
                                            void *userData)
{
    LogInfo<<"VideoDecodeCallback";
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
    LogInfo << "VideoDecodeInit";
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
    LogInfo << "VideoDecode";
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
    LogInfo << "GetFrames";
    MxBase::DeviceContext device;
    device.devId = DEVICE_ID;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->SetDevice(device);
    if (ret != APP_ERR_OK) {
        LogError << "SetDevice failed";
        return;
    }
    AVPacket pkt;
    while(!stopFlag){
        // 直接为一个已经分配好内存的指针或对象参数置为默认值，要求pkt的内存已经分配好了，如果为NULL，则此处会崩溃
        av_init_packet(&pkt);
        LogInfo << cnt;
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
        cnt++;
        // 原始帧数据被存储在Host侧
        MxBase::MemoryData streamData((void *)pkt.data, (size_t)pkt.size,
                                      MxBase::MemoryData::MEMORY_HOST_NEW, DEVICE_ID);
        ret = videoProcess->VideoDecode(streamData, VIDEO_HEIGHT, VIDEO_WIDTH, (void*)blockingQueue.get());
        if (ret != APP_ERR_OK) {
            LogError << "VideoDecode failed";
            return;
        }
        // 内部还是调用的av_init_packet,相当于先分配内存再设为默认值
        av_packet_unref(&pkt);
    }
    av_packet_unref(&pkt);
}

APP_ERROR VideoProcess::SaveResult(const std::shared_ptr<MxBase::MemoryData> resultInfo, const uint32_t frameId,
                                   std::vector<MxBase::ObjectInfo> &objInfos)
{
    LogInfo << "SaveResult";
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
        // 打印置信度最大推理结果
        LogInfo << "id: " << objInfos[i].classId << "; label: " << objInfos[i].className
                    << "; confidence: " << objInfos[i].confidence
                    << "; box: [ (" << objInfos[i].x0 << "," << objInfos[i].y0 << ") "
                    << "(" << objInfos[i].x1 << "," << objInfos[i].y1 << ") ]";

        int index = (int)objInfos[i].classId;
        const cv::Scalar color = cv::Scalar(color_num[index%200][0],color_num[index%200][1],color_num[index%200][2]); // 随机颜色
        const uint32_t thickness = 2;
        const uint32_t xOffset = 10;
        const uint32_t yOffset = 10;
        const uint32_t lineType = 8;
        const float fontScale = 1.0;
        // 在图像上绘制文字
        cv::putText(imgBgr, std::to_string((int)objInfos[i].classId), cv::Point(objInfos[i].x0 + xOffset, objInfos[i].y0 + yOffset),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, lineType);
        // 绘制矩形
        cv::rectangle(imgBgr,cv::Rect(objInfos[i].x0, objInfos[i].y0, 
            objInfos[i].x1 - objInfos[i].x0, objInfos[i].y1 - objInfos[i].y0),
            color, thickness);
        // 获取bounding box的中心位置
        center boxs = {(objInfos[i].x0+objInfos[i].x1)/2.0, (objInfos[i].y0+objInfos[i].y1)/2.0};
        // 保存每个车辆轨迹最新的20个bbox
        if(pts[index].size()>=20){
            pts[index].pop();
            pts[index].push(boxs);
        }
        else{
            pts[index].push(boxs);
        }
        std::vector<center> last_point = {};
        for(uint32_t j = 0; j < pts[index].size(); j++){
            if(pts[index].size()-j<=2){
                last_point.push_back(pts[index].front());
            }
            cv::circle(imgBgr, cv::Point(pts[index].front().x,pts[index].front().y), 1, color, 2);
            pts[index].push(pts[index].front());
            pts[index].pop();
        }
        // 不少于2个bbox的车辆轨迹可用于计数运算
        if(last_point.size()==2){
            if(intersect(last_point[1], last_point[0], line[0],line[1])){
                counter++;
                if(last_point[0].y>last_point[1].y)
                    counter_down++;
                else
                    counter_up++;
            }
        }
    }
    cv::line(imgBgr, cv::Point(line[0].x,line[0].y),cv::Point(line[1].x,line[1].y),  cv::Scalar(0, 255, 0),2);
    cv::putText(imgBgr,std::to_string(counter),cv::Point(20,90),0,0.8,cv::Scalar(0, 0, 255),2);
    cv::putText(imgBgr,std::to_string(counter_up),cv::Point(200,90),0,0.8,cv::Scalar(0, 255, 0),2);
    cv::putText(imgBgr,std::to_string(counter_down),cv::Point(450,90),0,0.8,cv::Scalar(255, 0, 0),2);
    frameIf.push(imgBgr);
    // 把Mat类型的图像矩阵保存为图像到指定位置。
    std::string fileName = "./result/result" + std::to_string(frameId+1) + ".jpg";
    cv::imwrite(fileName, imgBgr);
    ret = MxBase::MemoryHelper::MxbsFree(memoryDst);
    if(ret != APP_ERR_OK){
        LogError << "Fail to MxbsFree memory.";
        return ret;
    }
    return APP_ERR_OK;
}

void VideoProcess::GetResults(std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> blockingQueue,
                              std::shared_ptr<Yolov4Detection> yolov4Detection,
                              std::shared_ptr<VideoProcess> videoProcess, std::shared_ptr<ascendVehicleTracking::MOTConnection> tracker)
{
    LogInfo << "GetResults";
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
        ret = yolov4Detection->ResizeFrame(result, VIDEO_HEIGHT, VIDEO_WIDTH,resizeFrame);
        if (ret != APP_ERR_OK) {
            LogError << "Resize failed";
            return;
        }

        std::vector<MxBase::TensorBase> inputs = {};
        std::vector<MxBase::TensorBase> outputs = {};
        inputs.push_back(resizeFrame);
        // 推理
        ret = yolov4Detection->Inference(inputs, outputs);
        if (ret != APP_ERR_OK) {
            LogError << "Inference failed, ret=" << ret << ".";
            return;
        }
        std::vector<std::vector<MxBase::ObjectInfo>> objInfos;
        // 后处理
        ret = yolov4Detection->PostProcess(outputs, VIDEO_HEIGHT, VIDEO_WIDTH, objInfos);
        if (ret != APP_ERR_OK) {
            LogError << "PostProcess failed, ret=" << ret << ".";
            return;
        }
        // 追踪
        ret = tracker->ProcessSort(objInfos);
        if (ret != APP_ERR_OK) {
            LogError << "result failed ";
        }
        std::vector<MxBase::ObjectInfo> objInfos_ = {};
        ret = tracker->GettrackResult(objInfos_);
        if (ret != APP_ERR_OK) {
            LogError << "No tracker";
            return;
        }
        // 结果可视化
        ret = videoProcess->SaveResult(result, frameId, objInfos_);
        if (ret != APP_ERR_OK) {
            LogError << "Save result failed, ret=" << ret << ".";
            return;
        }
        frameId++;
        if(cnt == frameId){
            stopFlag = true;
            }
    }
}
// 获取每帧的可视化结果用于生成视频
std::queue<cv::Mat> VideoProcess::Getframes(){
    return frameIf;
}