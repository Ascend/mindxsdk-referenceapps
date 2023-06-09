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

std::string yoloModelPath = "model.om";
std::string yoloConfigPath = "configure.cfg";
std::string yoloLabelPath = "label.names";

std::string vehicleAttrModelPath = "model.om";
std::string vehicleAttrConfigPath = "configure.cfg";
std::string vehicleAttrLabelPath = "label.names";

std::string carPlateDetectModelPath = "model.om";
std::string carPlateDetectConfigPath = "configure.cfg";
std::string carPlateDetectLabelPath = "label.names";

std::string carPlateRecModelPath = "model.om";
std::string carPlateRecConfigPath = "configure.cfg";
std::string carPlateRecLabelPath = "label.names";

std::string pedestrianAttrModelPath = "model.om";
std::string pedestrianAttrConfigPath = "configure.cfg";
std::string pedestrianAttrLabelPath = "label.names";

std::string pedestrianFeatureModelPath = "model.om";

std::string faceLandmarkModelPath = "model.om";

std::string faceAttributeModelPath = "model.om";
std::string faceAttributeConfigPath = "configure.cfg";
std::string faceAttributeLabelPath = "label.names";

std::string faceFeatureModelPath = "model.om";

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

void resNetFeatureProcess(std::vector<MxBase::Tensor> &inferOutputs, std::vector<float> &feature, bool isSigmoid)
{
    const int FEATURE_SIZE = 4;
    if (inferOutputs.empty())
    {
        printf("result Infer failed with empty output...\n");
        retrun;
    }

    size_t featureSize = inferOutputs[0].GetByteSize() / FEATURE_SIZE;
    float *castData = static_cast<float *>(inferOutputs[0].GetData());

    for (size_t i = 0; i < featureSize; i++)
    {
        features.push_back(activateOutput(castData[i], isSigmoid));
    }
    return;
}

void yoloImagePreProcessor()
{
}

template <size_t NUM>
void dispatchParallelPipeline(int batch, tf::Pipeline<tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>> *&pl,
                              MxBase::ImageProcessor *&imageProcessor,
                              MultiObjectTracker *&multiObjectTracker,
                              MxBase::Model *&yoloModel,
                              MxBase::Yolov3PostProcess *&yoloPostProcessor,
                              MxBase::Model *&vehicleAttrModel,
                              ResnetAttributePostProcess *&vehicleAttrPostprocessor,
                              MxBase::Model *&carPlateDetectionModel,
                              SsdVggPostProcess *&carPlateDetectPostProcessor,
                              MxBase::Model *&carPlateRecognitionModel,
                              CarPlateRecognitionPostProcess *&carPlateRecognitionPostProcessor,
                              MxBase::Model *&pedestrianAttrModel,
                              ResnetAttributePostProcess *&pedestrianAttrPostProcessor,
                              MxBase::Model *&pedestrianFeatureModel,
                              MxBase::Model *&faceLandmarkModel,
                              FaceLandmarkPostProcess *&faceLandmarkPostProcessor,
                              FaceAlignment *&faceAlignmentProcessor,
                              MxBase::Model *&faceAttributeModel,
                              ResnetAttributePostProcess *&faceAttributeProcessor,
                              MxBase::Model *&faceFeatureModel,
                              MxBase::BlockingQueue<FrameImage> &decodedFrameQueue,
                              std::array<std::pair<FrameImage, std::vector<MxBase::ObjectInfo>>, NUM> &selectedObjectBuffer,
                              std::array<std::vector<PreprocessedImage>, NUM> &vehicleAttrInputImageBuffer,
                              std::array<std::vector<PreprocessedImage>, NUM> &carPlateDetectionInputImageBuffer,
                              std::array<std::vector<PreprocessedImage>, NUM> &carPlateRecognitionInputImageBuffer,
                              std::array<std::vector<PreprocessedImage>, NUM> &pedestrianAttrInputImageBuffer,
                              std::array<std::vector<PreprocessedImage>, NUM> &pedestrianFeatureInputImageBuffer,
                              std::array<std::vector<PreprocessedImage>, NUM> &faceLandmarkInputImageBuffer,
                              std::array<std::vector<PreprocessedImage>, NUM> &faceAlignedImageBuffer,
                              std::array<std::vector<PreprocessedImage>, NUM> &faceFeatureInputImageBuffer,
                              std::array<std::vector<MxBase::Tensor>, NUM> &outputs,
                              uint32_t &deviceID, std::array<FrameImage, NUM> &buffer,
                              std::array<MxBase::Image, NUM> &resizedImageBuffer, tf::Executor &executor)
{
    pl = new tf::Pipeline{numLines,
                          tf::Pipe<std::function<void(tf::Pipeflow &)>>{tf::PipeType::SERIAL,
                                                                        [&, batch](tf::Pipeflow &pf) {

                                                                                                                                                }

                          },

                          tf::Pipe<std::function<void(tf::Pipeflow &)>>{tf::PipeType::SERIAL,
                                                                        [&, batch](tf::Pipeflow &pf) {

                                                                        }

                          },

                          tf::Pipe<std::function<void(tf::Pipeflow &)>>{tf::PipeType::SERIAL,
                                                                        [&, batch](tf::Pipeflow &pf) {

                                                                        }

                          },

                          tf::Pipe<std::function<void(tf::Pipeflow &)>>{tf::PipeType::SERIAL,
                                                                        [&, batch](tf::Pipeflow &pf) {

                                                                        }

                          },

                          tf::Pipe<std::function<void(tf::Pipeflow &)>>{tf::PipeType::SERIAL,
                                                                        [&, batch](tf::Pipeflow &pf) {

                                                                        }

                          },

                          tf::Pipe<std::function<void(tf::Pipeflow &)>>{tf::PipeType::SERIAL,
                                                                        [&, batch](tf::Pipeflow &pf) {

                                                                        }

                          },

                          tf::Pipe<std::function<void(tf::Pipeflow &)>>{tf::PipeType::SERIAL,
                                                                        [&, batch](tf::Pipeflow &pf) {

                                                                        }

                          },

                          tf::Pipe<std::function<void(tf::Pipeflow &)>>{tf::PipeType::SERIAL,
                                                                        [&, batch](tf::Pipeflow &pf) {

                                                                        }

                          },

                          tf::Pipe<std::function<void(tf::Pipeflow &)>>{tf::PipeType::SERIAL,
                                                                        [&, batch](tf::Pipeflow &pf) {

                                                                        }

                          },

                          tf::Pipe<std::function<void(tf::Pipeflow &)>>{tf::PipeType::SERIAL,
                                                                        [&, batch](tf::Pipeflow &pf) {

                                                                        }

                          },

                          tf::Pipe<std::function<void(tf::Pipeflow &)>>{tf::PipeType::SERIAL,
                                                                        [&, batch](tf::Pipeflow &pf) {

                                                                        }

                          },

                          tf::Pipe<std::function<void(tf::Pipeflow &)>>{tf::PipeType::SERIAL,
                                                                        [&, batch](tf::Pipeflow &pf) {

                                                                        }

                          },

                          tf::Pipe<std::function<void(tf::Pipeflow &)>>{tf::PipeType::SERIAL,
                                                                        [&, batch](tf::Pipeflow &pf) {

                                                                        }

                          }};
}

int main(int argc, char *argv[])
{
    MxBase::MxInit();

    av_register_all();

    avformat_network_init();

    initResources();
    sleep(5);

    std::vecor<std::string> filePaths(numChannel);

    std::fill(filePaths.begin(), filePaths.end(), "../test.264");

    AVFormatContext *pFormatCtx[numChannel];
    AVPacket pkt[numChannel];
    std::vector<uint32_t> frameIDs(numChannel, deviceID);

    MxBase::VideoDecodeConfig config;
    MxBase::VideoDecodeCallBack cPtr = CallBackVdec;
    config.width = 1920;
    config.height = 1080;
    config.callbackFunc = cPtr;
    config.skipInterval = 3;
    config.inputVideoFormat = MxBase::StreamFormat::H264_MAIN_LEVEL;

    MxBase::VideoDecoder *videoDecoder[numChannel];

    tf::Executor executor(256);

    tf::Taskflow taskflow;
    tf::Task init = task.emplace([]()
                                 { print("ready\n"); })
                        .name("starting pipeline");
    tf::Task stop = task.emplace([]()
                                 { print("stopped\n"); })
                        .name("pipeline stopped");

    for (size_t i = 0; i < numChannel; ++i)
    {
        executor.async([&, i]()
                       {
            pFormatCtx[i] = nullptr;
            videoDecoder[i] = new MxBase::VideoDecoder(config, deviceIDs[i], i);

            StreamPull(pFormatCtx[i], filePaths[i]);
            if (pFormatCtx[i] == nullptr) {
                printf("is nullptr\n");
            }
            VideoDecode(pFormatCtx[i], pkt[i], decodedFrameQueueList[i % numWoker], i, frameIDs[i], deviceIDs[i], videoDecoder[i], decodeEOF[i], executor);

            delete videoDecoder[i]; });
    }

    tf::Task detectTask[numWoker] = {};
    std::array<std::array<FrameImage, numLines>, numWoker> yoloFrameBuffer;
    std::array<std::array<MxBase::Image, numLines>, numWoker> yoloResizedImageBuffer;
    std::array<std::array<std::pair<FrameImage, std::vector<MxBase::ObjectInfo>>, numLines>, numWoker> selectedObjectBuffer;

    std::array<std::array<std::vector<PreprocessedImage>, numLines>, numWoker> vehicleAttrInputImageBuffer;
    std::array<std::array<std::vector<PreprocessedImage>, numLines>, numWoker> carPlateDetectionInputImageBuffer;
    std::array<std::array<std::vector<PreprocessedImage>, numLines>, numWoker> carPlateRecognitionInputImageBuffer;
    std::array<std::array<std::vector<PreprocessedImage>, numLines>, numWoker> pedestrianAttrInputImageBuffer;
    std::array<std::array<std::vector<PreprocessedImage>, numLines>, numWoker> pedestrianFeatureInputImageBuffer;
    std::array<std::array<std::vector<PreprocessedImage>, numLines>, numWoker> faceAttrInputImageBuffer;
    std::array<std::array<std::vector<PreprocessedImage>, numLines>, numWoker> faceAlignedImageBuffer;
    std::array<std::array<std::vector<PreprocessedImage>, numLines>, numWoker> faceFeatureInputImageBuffer;

    std::array<std::array<std::vector<MxBase::Tensor>, numLines>, numWoker> yoloResults;
    tf::Pipeline<tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>,
                 tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>,
                 tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>,
                 tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>,
                 tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>,
                 tf::Pipe<std::function<void(tf::Pipeflow &)>>, tf::Pipe<std::function<void(tf::Pipeflow &)>>,
                 tf::Pipe<std::function<void(tf::Pipeflow &)>>>
        fullPipelines[numWoker];

    for (int workerIndex = 0; wokerIndex < numWoker; wokerIndex++)
    {
        printf("====================dispatch yolo pipeline=========================\n");
        dispatchParallelPipeline(workerIndex, fullPipelines[workerIndex],
                                 imageProcessors[workerIndex],
                                 multiObjectTrackers[wokerIndex],
                                 yoloModels[wokerIndex],
                                 yoloPostProcessors[wokerIndex],
                                 vehicleAttrModels[workerIndex],
                                 vehicleAttrPostProcessors[wokerIndex],
                                 carPlateDetectModels[wokerIndex],
                                 carPlateDetectPostProcessors[wokerIndex],
                                 carPlateRecModels[wokerIndex],
                                 carPlateRecPostProcessors[wokerIndex],
                                 pedestrianAttrModels[wokerIndex],
                                 pedestrianAttrPostProcessors[wokerIndex],
                                 pedestrianFeatureModels[wokerIndex],
                                 faceLandmarkModels[wokerIndex],
                                 faceLandmarkPostProcessors[wokerIndex],
                                 faceAlignmentProcessors[wokerIndex],
                                 faceAttributeModels[wokerIndex],
                                 faceAttributeProcessors[wokerIndex],
                                 faceFeatureModels[wokerIndex],
                                 decodedFrameQueueList[wokerIndex],
                                 selectedObjectBuffer[wokerIndex],
                                 vehicleAttrInputImageBuffer[wokerIndex],
                                 carPlateDetectionInputImageBuffer[wokerIndex],
                                 carPlateRecognitionInputImageBuffer[wokerIndex],
                                 pedestrianAttrInputImageBuffer[wokerIndex],
                                 pedestrianFeatureInputImageBuffer[wokerIndex],
                                 faceAttrInputImageBuffer[wokerIndex],
                                 faceAlignedImageBuffer[wokerIndex],
                                 faceFeatureInputImageBuffer[wokerIndex],
                                 yoloResults[wokerIndex], deviceIDs[wokerIndex],
                                 yoloFrameBuffer[wokerIndex],
                                 yoloResizedImageBuffer[wokerIndex],
                                 executor);

        detectTask[wokerIndex] = taskflow.composed_of(*fullPipelines[wokerIndex]).name("pipeline1");
        init.precede(detectTask[wokerIndex]);
        detectTask[wokerIndex].precede(stop);
    }

    auto start = std::chrono::steady_clock::now();
    printf("Tasks dispatched\n");
    taskflow.dump(std::cout);
    executor.run(taskflow);

    executor.wait_for_all();

    printf("All tasks finished\n");
    auto end = std::chrono::steady_clock::now();

    printf("numChannel: %zu, numWoker: %zu\n", numChannel, numWoker);
    printf("Elapsed time in microseconds: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}