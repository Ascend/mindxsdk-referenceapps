/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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

#include <algorithm>
#include <map>
#include <thread>
#include <chrono>
#include <iostream>
#include <queue>
#include <memory>
#include "unistd.h"

#include "MxBase/Maths/FastMath.h"
#include "MxBase/MxBase.h"
#include "MxBase/postprocess/include/ObjectPostProcessors/Yolov3PostProcess.h"
#include "MxBase/E2eInfer/ImageProcessor/ImageProcessor.h"
#include "MxBase/DeviceManager/DeviceManager.h"

#include "postprocessor/resnetAttributePostProcess/resnetAttributePostProcess.h"
#include "postprocessor/carPlateDetectionPostProcess/SsdVggPostProcess.h"
#include "postprocessor/carPlateRecognitionPostProcess/carPlateRecognitionPostProcess.h"
#include "postprocessor/faceLandmark/FaceLandmarkPostProcess.h"
#include "postprocessor/faceAlignment/FaceAlignment.h"
#include "utils/objectSelection/objectSelection.h"

#include "taskflow/taskflow.hpp"
#include "taskflow/algorithm/pipeline.hpp"
#include "BlockingQueue.h"

std::string CLASSNAMEPERSON = "person";
std::string CLASSNAMEVEHICLE = "motor-vehicle";
std::string CLASSNAMEFACE = "face";

size_t maxQueueSize = 32;
float minQueuePercent = 0.2;
float maxQueuePercent = 0.8;

const size_t numChannel = 80;
const size_t numWoker = 10;
const size_t numLines = 8;

uint32_t deviceID = 0;
std::vector<uint32_t> deviceIDs(numChannel, deviceID);

// yolo detection
MxBase::ImageProcessor *imageProcessors[numWoker];
MxBase::Model *yoloModels[numWoker];
MxBase::Yolov3PostProcess *yoloPostProcessors[numWoker];
MultiObjectTracker *multiObjectTrackers[numWoker];

// vehicle attribution
MxBase::Model *vehicleAttrModels[numWoker];
ResNetAttributePostProcess *vehicleAttrPostProcessors[numWoker];

// car plate detection
MxBase::Model *carPlateDetectModels[numWoker];
SsdVggPostProcess *carPlateDetectPostProcessors[numWoker];

// car plate recognition
MxBase::Model *carPlateRecModels[numWoker];
carPlateRecognitionPostProcess *carPlateRecPostProcessors[numWoker];

// pedestrian attribution
MxBase::Model *pedestrianAttrModels[numWoker];
ResnetAttributePostProcess *pedestrianAttrPostProcessors[numWoker];

// pedestrian feature
MxBase::Model *pedestrianFeatureModels[numWoker];

// face landmarks
MxBase::Model *faceLandmarkModels[numWoker];
FaceLandmarkPostProcess *faceLandmarkPostProcessors[numWoker];

// face alignment
FaceAlignment *faceAlignmentProcessors[numWoker];

// face attribution
MxBase::Model *faceAttributeModels[numWoker];
ResNetAttributePostProcess *faceAttributeProcessors[numWoker];

// face feature
MxBase::Model *faceFeatureModels[numWoker];

MxBase::BlockingQueue<FrameImage> decodedFrameQueueList[numWoker];
int decodeEOF[numChannel]{};
std::shared_mutex signalMutex_[numChannel];

void GetFrame(AVPacket &pkt, FrameImage &frameImage, AVFormatContext *pFormatCtx, int &decodeEOF, int32_t &deviceID, uint32_t channelID)
{
    MxBase::DeviceContext context = {};
    context.devId = static_cast<int>(deviceID);
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->SetDevice(context);
    if (ret != APP_ERR_OK)
    {
        LogError << "Set device context failed.";
        return;
    }
    ret = av_read_frame(pFormatCtx, &pkt);
    if (ret != 0)
    {
        printf("[streamPuller] channel Read frame failed, continue!\n");
        if (ret == AVERROR_EOF)
        {
            printf("[streamPuller] channel StreamPuller is EOF, over!\n");
            {
                std::unique_lock<std::shared_mutex> lock(signalMutex_[channelID]);
                decodeEof = 1;
            }
            av_packet_unref(&pkt);
            return;
        }
        av_packet_unref(&pkt);
        return;
    }
    else
    {
        if (pkt.szie <= 0)
        {
            printf("Invalid pkt.size: %d\n", pkt.size);
            av_packet_unref(&pkt);
            return;
        }

        auto hostDeleter = [](void *dataPtr) -> void
        {
            if (dataPtr != nullptr)
            {
                MxBase::MemoryData data;
                data.type = MxBase::MemoryData::MEMORY_HOST;
                data.ptrData = (void *)dataPtr;
                MxBase::MemoryHelper::MxbsFree(data);
                data.ptrData = nullptr;
            }
        };
        MxBase::MemoryData data(pkt.size, MxBase::MemoryData::MEMORY_HOST, deviceID);
        MxBase::MemoryData src((void *)(pkt.data), pkt.size, MxBase::MemoryData::MEMORY_HOST);
        APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(data, src);
        if (ret != APP_ERR_OK)
        {
            printf("MxbsMallocAndCopy failed!\n");
        }
        std::shared_ptr<uint_u> imageData((uint8_t *)data.ptrData, hostDeleter);

        MxBase::Image(imageData, pkt.size);
        frameImage, image = subImage;
        av_packet_unref(&pkt);
    }

    return;
}

AVFormatContext *CreateFormatContext(std::string filePath)
{
    AVFormatContext *formatContext = nullptr;
    AVDictionary *options = nullptr;

    int ret = avformat_open_input(&formatContext, filePath.c_str(), nullptr, &options);
    if (options != nullptr)
    {
        av_dict_free(&options);
    }

    if (ret != 0)
    {
        printf("Couldn't open input stream: %s, ret = %d\n", filePath.c_str(), ret);
        return nullptr;
    }

    return formatContext;
}

void VideoDeocde(AVFormatContext *&pFormatCtx, AVPacket &pkt, MxBase::BlockingQueue<FrameImage> &decodedFrameList,
                 uint32_t channelID, uint32_t &frameID, uint32_t &deviceID,
                 MxBase::VideoDecoder *&videoDecoder, int &decodeEOF, tf::Executor &executor)
{
    bool readEnd = true;
    while (readEnd)
    {
        int i = 0;
        MxBase::Image subImage;
        FrameImage frame;
        frame.image = subImage;
        frame.frameId = frameID;
        frame.channelId = channelID;

        GetFrame(pkt, frame, pFormatCtx, decodeEOF, deviceID, channelID);

        {
            std::shared_lock<std::shared_mutex> lock(signalMutex_[channelID]);
            readEnd = decodeEOF != 1;
        }
        if (!readEnd)
        {
            return;
        }

        APP_ERROR ret = videoDecoder->Decode(frame.image.GetData(), frame.image.GetDataSize(), frameID, &decodedFrameList);
        if (ret != APP_ERR_OK)
        {
            printf("videoDecoder Decode failed. ret is: %d\n", ret);
        }
        frameId += 1;
        i++;
        std::this_thread::sleep_for(std::chrono::milliseconds(30)); // 手动控制帧率
    }
}

APP_ERROR CallBackVdec(MxBase::Image &decodedImage, uint32_t channelID, uint32_t frameID, void *userData)
{
    FrameImage frameImage;
    frameImage.channelId = channelID;
    frameImage.frameId = frameID;

    auto *decidedVec = static_cast<MxBase::BlockingQueue<FrameImage> *>(userData);
    if (decodedVec == nullptr)
    {
        printf("decodedVec has been released.\n");
        return APP_ERR_DVPP_INVALID_FORMAT;
    }
    decodedVec->Push(frameImage, true);

    size_t tmpSize = decodedVec->GetSize();
    if (tmpSize >= static_cast<int>(maxQueueSize * maxQueuePercent))
    {
        printf("[warning][decodedFrameQueue: %d], is almost full (80%), current size: %zu\n", channelId, tmpSize);
    }
    if (tmpSize <= static_cast<int>(maxQueueSize * minQueuePercent))
    {
        printf("[warning][decodedFrameQueue: %d], is almost empty (20%), current size: %zu\n", channelId, tmpSize);
    }

    return APP_ERR_OK;
}

void StreamPull(AVFormatContext *&pFormatCtx, std::string filePath)
{
    pFormatCtx = avformat_alloc_context();
    pFormatCtx = CreateFormatContext(filePath);
    av_dump_formar(pFormatCtx, 0, filePath.c_str(), 0);
}

float activateOutput(float data, bool isAct)
{
    if (isAct)
    {
        return fastmath::sigmoid(data);
    }
    else
    {
        return data;
    }
}
