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

#include "VideoGestureReasoner.h"
#include "../Util/Util.h"
#include "MxBase/DeviceManager/DeviceManager.h"

// init static variable
bool VideoGestureReasoner::forceStop = false;

APP_ERROR VideoGestureReasoner::Init(const ReasonerConfig &initConfig)
{
    LogDebug << "Init VideoGestureReasoner start.";
    APP_ERROR ret;

    ret = CreateStreamPullerAndVideoDecoder(initConfig);
    if (ret != APP_ERR_OK) {
        LogError << "CreateStreamPullerAndVideoDecoder failed.";
        return ret;
    }

    ret = CreateFrameSkippingSampling(initConfig);
    if (ret != APP_ERR_OK) {
        LogError << "CreateFrameSkippingSampling failed.";
        return ret;
    }

    ret = CreateImageResizer(initConfig);
    if (ret != APP_ERR_OK) {
        LogError << "CreateImageResizer failed.";
        return ret;
    }

    ret = CreateResnetDetector(initConfig);
    if (ret != APP_ERR_OK) {
        LogError << "CreateResnetDetector failed.";
        return ret;
    }

    this->deviceId = initConfig.deviceId;
    this->resnetModelWidth = initConfig.resnetModelWidth;
    this->resnetModelHeight = initConfig.resnetModelHeight;
    this->popDecodeFrameWaitTime = initConfig.popDecodeFrameWaitTime;
    this->maxDecodeFrameQueueLength = initConfig.maxDecodeFrameQueueLength;

    this->stopFlag = false;
    return APP_ERR_OK;
}

void VideoGestureReasoner::Process()
{
    std::vector<std::thread> videoProcessThreads;

    auto decodeFrameQueue = std::make_shared<BlockingQueue<std::shared_ptr<void>>>(maxDecodeFrameQueueLength);
    std::thread getDecodeVideoFrame(GetDecodeVideoFrame,
                                    streamPullers[0],
                                    videoDecoders[0],
                                    decodeFrameQueue, this);

    // save
    videoProcessThreads.push_back(std::move(getDecodeVideoFrame));
    decodeFrameQueueMap.insert(std::pair<int,
                               std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>>(0, decodeFrameQueue));

    std::thread getDetectionResult(GetDetectionResult,
                                   resnetModelWidth,
                                   resnetModelHeight,
                                   popDecodeFrameWaitTime, this);
    videoProcessThreads.push_back(std::move(getDetectionResult));

    while (!stopFlag) {
        bool allVideoDataPulledAndDecoded = true;

        if (streamPullers[0]->stopFlag) {
            if (!videoDecoders[0]->stopFlag) {
                LogDebug << "video frame decoded and no fresh data, quit video decoder. ";
                videoDecoders[0]->stopFlag = true;
            }
        } else {
            allVideoDataPulledAndDecoded = false;
        }

        bool allVideoDataProcessed = !Util::IsExistDataInQueueMap(decodeFrameQueueMap);
        if (allVideoDataPulledAndDecoded && allVideoDataProcessed) {
            LogDebug << "all decoded frame detected and no fresh data, quit image resizer and resnet detector";
            imageResizer->stopFlag = true;
            resnetDetector->stopFlag = true;
        }

        if (allVideoDataPulledAndDecoded && imageResizer->stopFlag && resnetDetector->stopFlag) {
            LogDebug << "Both of stream puller, video decoder, image resizer and resnet detector quit, main quit";
            stopFlag = true;
        }

        // force stop case
        if (VideoGestureReasoner::forceStop) {
            LogInfo << "Force stop VideoGestureReasoner.";
            stopFlag = true;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(delayTime));
    }

    // threads join
    for (auto & videoProcessThread : videoProcessThreads) {
        videoProcessThread.join();
    }
}

APP_ERROR VideoGestureReasoner::DeInit()
{
    APP_ERROR ret;

    ret = DestroyStreamPullerAndVideoDecoder();
    if (ret != APP_ERR_OK) {
        LogError << "DestroyStreamPullerAndVideoDecoder failed.";
        return ret;
    }

    ret = DestroyFrameSkippingSampling();
    if (ret != APP_ERR_OK) {
        LogError << "DestroyFrameSkippingSampling failed.";
        return ret;
    }

    ret = DestroyImageResizer();
    if (ret != APP_ERR_OK) {
        LogError << "DestroyImageResizer failed.";
        return ret;
    }

    ret = DestroyResnetDetector();
    if (ret != APP_ERR_OK) {
        LogError << "DestroyresnetDetector failed.";
        return ret;
    }

    ClearData();

    this->stopFlag = true;
    return APP_ERR_OK;
}

/// ========== static Method ========== ///
void VideoGestureReasoner::GetDecodeVideoFrame(const std::shared_ptr<AscendStreamPuller::StreamPuller> &streamPuller,
                                               const std::shared_ptr<AscendVideoDecoder::VideoDecoder> &videoDecoder,
                                               const std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> &decodeFrameQueue,
                                               const VideoGestureReasoner *videoGestureReasoner)
{
    // set device
    MxBase::DeviceContext device;
    device.devId = videoGestureReasoner->deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->SetDevice(device);
    if (ret != APP_ERR_OK) {
        LogError << "SetDevice failed";
        return;
    }

    while (true) {
        if (videoGestureReasoner->stopFlag) {
            LogDebug << "stop video stream pull and video frame decode";
            streamPuller->stopFlag = true;
            videoDecoder->stopFlag = true;
            break;
        }

        if (streamPuller->stopFlag) {
            LogDebug << "no video frame to pull and all pulled video frame decoded. quit!";
            break;
        }

        // video stream pull
        auto videoFrameData = streamPuller->GetNextFrame();
        if (videoFrameData.size == 0) {
            LogDebug << "empty video frame, not need decode, continue!";
            continue;
        }
        videoDecoder->Decode(videoFrameData,
                             streamPuller->GetFrameInfo().width,
                             streamPuller->GetFrameInfo().height,
                             decodeFrameQueue.get());
    }
}

void VideoGestureReasoner::GetDetectionResult(const uint32_t &modelWidth,
                                              const uint32_t &modelHeight,
                                              const uint32_t &popDecodeFrameWaitTime,
                                              const VideoGestureReasoner *videoGestureReasoner)
{
    // set device
    MxBase::DeviceContext device;
    device.devId = videoGestureReasoner->deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->SetDevice(device);
    if (ret != APP_ERR_OK) {
        LogError << "SetDevice failed";
        return;
    }

    auto imageResizer = videoGestureReasoner->imageResizer;
    auto resnetDetector = videoGestureReasoner->resnetDetector;
    auto decodeFrameQueueMap = videoGestureReasoner->decodeFrameQueueMap;
    auto videoFrameInfos = videoGestureReasoner->videoFrameInfos;
    auto frameSkippingSampling = videoGestureReasoner->frameSkippingSampling;
    while (true) {
        if (videoGestureReasoner->stopFlag) {
            LogDebug << "stop image resize and resnet detect";
            imageResizer->stopFlag = true;
            resnetDetector->stopFlag = true;
            break;
        }

        if (imageResizer->stopFlag && resnetDetector->stopFlag) {
            LogDebug << "no image need to resize and all image detected. quit!";
            break;
        }

        if (!Util::IsExistDataInQueueMap(decodeFrameQueueMap)) {
            continue;
        }

        std::_Rb_tree_const_iterator<std::pair<const int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>>> iter;
        for (iter = decodeFrameQueueMap.begin(); iter != decodeFrameQueueMap.end(); iter++) {
            auto rtspIndex = iter->first;
            auto decodeFrameQueue = iter->second;
            if (decodeFrameQueue->IsEmpty()) {
                continue;
            }

            auto startTime = std::chrono::high_resolution_clock::now();
            ret = frameSkippingSampling->Process();
            if (ret != APP_ERR_OK) {
                LogError << "FrameSkippingSampling failed";
                continue;
            }
            auto endTime = std::chrono::high_resolution_clock::now();
            double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
            LogInfo << "frameSkippingSampling time: " << costMs;

            // get decode frame data
            std::shared_ptr<void> data = nullptr;
            ret = decodeFrameQueue->Pop(data, popDecodeFrameWaitTime);
            if (ret != APP_ERR_OK) {
                LogError << "Pop failed";
                continue;
            }
            auto decodeFrame = std::make_shared<MxBase::MemoryData>();
            decodeFrame = std::static_pointer_cast<MxBase::MemoryData>(data);

            if (frameSkippingSampling->stopFlag) {
                APP_ERROR ret = MxBase::DeviceManager::GetInstance()->SetDevice(device);
                LogError << "resize frame and iter :";
                // resize frame
                MxBase::DvppDataInfo resizeFrame = {};
                auto startTime = std::chrono::high_resolution_clock::now();
                AscendImageResizer::ImageResizerParma imageInitParma;
                imageInitParma.originHeight = videoFrameInfos[rtspIndex].height;
                imageInitParma.originWidth = videoFrameInfos[rtspIndex].width;
                imageInitParma.resizeHeight = modelHeight;
                imageInitParma.resizeWidth = modelWidth;

                ret = imageResizer->ResizeFromMemory(*decodeFrame, imageInitParma, resizeFrame);
                if (ret != APP_ERR_OK) {
                    LogError << "Resize image failed, ret = " << ret << " " << GetError(ret);
                    continue;
                }
                auto endTime = std::chrono::high_resolution_clock::now();
                double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
                LogInfo << "resize frame time: " << costMs;

                // resnet detect
                std::vector<std::vector<MxBase::ClassInfo>> objInfos;
                startTime = std::chrono::high_resolution_clock::now();
                ret = resnetDetector->Detect(resizeFrame, objInfos,
                                             videoFrameInfos[rtspIndex].width,
                                             videoFrameInfos[rtspIndex].height);
                if (ret != APP_ERR_OK) {
                    LogError << "Resnet detect image failed, ret = " << ret << " " << GetError(ret) << ".";
                    continue;
                }
                endTime = std::chrono::high_resolution_clock::now();
                costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
                LogInfo << "resnet detect time: " << costMs;

                // save detect result
                ret = Util::SaveResult(decodeFrame, resizeFrame.frameId, objInfos,
                                       videoFrameInfos[rtspIndex].width, videoFrameInfos[rtspIndex].height, rtspIndex);
                if (ret != APP_ERR_OK) {
                    LogError << "Save result failed, ret=" << ret << ".";
                    return;
                }
            }
        }
    }
}

/// ========== private Method ========== ///
APP_ERROR VideoGestureReasoner::CreateStreamPullerAndVideoDecoder(const ReasonerConfig &config)
{
    auto rtspList = config.rtspList;

    APP_ERROR ret;
    AscendStreamPuller::VideoFrameInfo videoFrameInfo;
    AscendVideoDecoder::DecoderInitParam decoderInitParam = {};

    auto streamPuller = std::make_shared<AscendStreamPuller::StreamPuller>();
    auto videoDecoder = std::make_shared<AscendVideoDecoder::VideoDecoder>();

    ret = streamPuller->Init(rtspList[0], config.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Init " << " StreamPuller failed, stream name: " << rtspList[0];
        return ret;
    }
    videoFrameInfo = streamPuller->GetFrameInfo();

    Util::InitVideoDecoderParam(decoderInitParam, config.deviceId, config.baseVideoChannelId, videoFrameInfo);
    ret = videoDecoder->Init(decoderInitParam);
    if (ret != APP_ERR_OK) {
        LogError << "Init " << " VideoDecoder failed";
        return ret;
    }

    // save
    streamPullers.push_back(streamPuller);
    videoDecoders.push_back(videoDecoder);
    videoFrameInfos.push_back(videoFrameInfo);

    return APP_ERR_OK;
}

APP_ERROR VideoGestureReasoner::CreateFrameSkippingSampling(const ReasonerConfig &config)
{
    APP_ERROR ret;
    frameSkippingSampling = std::make_shared<AscendFrameSkippingSampling::FrameSkippingSampling>();
    ret = frameSkippingSampling->Init(config.maxSamplingInterval, config.samplingInterval, config.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Init SamplingInterval failed";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR VideoGestureReasoner::CreateImageResizer(const ReasonerConfig &config)
{
    imageResizer = std::make_shared<AscendImageResizer::ImageResizer>();

    APP_ERROR ret = imageResizer->Init(config.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Init image resizer failed";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR VideoGestureReasoner::CreateResnetDetector(const ReasonerConfig &config)
{
    resnetDetector = std::make_shared<AscendResnetDetector::ResnetDetector>();

    AscendResnetDetector::ResnetInitParam resnetInitParam;
    Util::InitResnetParam(resnetInitParam, config.deviceId, config.resnetLabelPath, config.resnetModelPath);
    APP_ERROR ret = resnetDetector->Init(resnetInitParam);
    if (ret != APP_ERR_OK) {
        LogError << "Init resnet detector failed.";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR VideoGestureReasoner::DestroyStreamPullerAndVideoDecoder()
{
    APP_ERROR ret;
    // deinit video decoder
    ret = videoDecoders[0]->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "Deinit " << " VideoDecoder failed";
        return ret;
    }

    // deinit stream puller
    ret = streamPullers[0]->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "Deinit "  << " StreamPuller failed.";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR VideoGestureReasoner::DestroyFrameSkippingSampling()
{
    APP_ERROR ret = frameSkippingSampling->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "FrameSkippingSampling DeInit failed.";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR VideoGestureReasoner::DestroyImageResizer()
{
    APP_ERROR ret = imageResizer->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "ImageResizer DeInit failed.";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR VideoGestureReasoner::DestroyResnetDetector()
{
    APP_ERROR ret = resnetDetector->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "ResnetDetector DeInit failed.";
        return ret;
    }
    return APP_ERR_OK;
}

const void VideoGestureReasoner::ClearData()
{
    // stop and clear queue
    std::_Rb_tree_const_iterator<std::pair<const int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>>> iter;
    for (iter = decodeFrameQueueMap.begin(); iter != decodeFrameQueueMap.end(); iter++) {
        iter->second->Stop();
        iter->second->Clear();
    }
    decodeFrameQueueMap.clear();
    videoFrameInfos.clear();
    videoDecoders.clear();
    streamPullers.clear();
}

