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


#include "MultiChannelVideoReasoner.h"
#include "../Util/Util.h"

#include "MxBase/DeviceManager/DeviceManager.h"

APP_ERROR MultiChannelVideoReasoner::Init(const ReasonerConfig &initConfig)
{
    LogDebug << "Init MultiChannelVideoReasoner start.";
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

void MultiChannelVideoReasoner::Process()
{
    std::vector<std::thread> videoProcessThreads;
    for (uint32_t i = 0; i < streamPullers.size(); i++) {
        auto decodeFrameQueue = std::make_shared<BlockingQueue<std::shared_ptr<void>>>(maxDecodeFrameQueueLength);
        std::thread getDecodeVideoFrame(GetDecodeVideoFrame, streamPullers[i], videoDecoders[i], decodeFrameQueue, this);

        // save
        videoProcessThreads.push_back(std::move(getDecodeVideoFrame));
        decodeFrameQueueMap.insert(std::pair<int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>>(i, decodeFrameQueue));
    }

    std::thread getMultiChannelDetectionResult(GetMultiChannelDetectionResult, resnetModelWidth, resnetModelHeight, popDecodeFrameWaitTime, this);
    videoProcessThreads.push_back(std::move(getMultiChannelDetectionResult));

    while (!stopFlag) {
        bool allVideoDataPulledAndDecoded = true;

        for (uint32_t i = 0; i < streamPullers.size(); i++) {
            if (streamPullers[i]->stopFlag) {
                if (!videoDecoders[i]->stopFlag) {
                    LogDebug << "all video frame decoded and no fresh data, quit video decoder " + std::to_string(i) + ".";
                    videoDecoders[i]->stopFlag = true;
                }
            } else {
                allVideoDataPulledAndDecoded = false;
            }
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
        if (MultiChannelVideoReasoner::forceStop) {
            LogInfo << "Force stop MultiChannelVideoReasoner.";
            stopFlag = true;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    // threads join
    for (auto & videoProcessThread : videoProcessThreads) {
        videoProcessThread.join();
    }
}

APP_ERROR MultiChannelVideoReasoner::DeInit()
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
void MultiChannelVideoReasoner::GetDecodeVideoFrame(
        const std::shared_ptr<AscendStreamPuller::StreamPuller> &streamPuller,
        const std::shared_ptr<AscendVideoDecoder::VideoDecoder> &videoDecoder,
        const std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> &decodeFrameQueue,
        const MultiChannelVideoReasoner* multiChannelVideoReasoner)
{
    // set device
    MxBase::DeviceContext device;
    device.devId = multiChannelVideoReasoner->deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->SetDevice(device);
    if (ret != APP_ERR_OK) {
        LogError << "SetDevice failed";
        return;
    }

    while (true) {
        if (multiChannelVideoReasoner->stopFlag) {
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

        videoDecoder->Decode(videoFrameData, streamPuller->GetFrameInfo().width, streamPuller->GetFrameInfo().height, decodeFrameQueue.get());

    }
}

void MultiChannelVideoReasoner::GetMultiChannelDetectionResult(const uint32_t &modelWidth, const uint32_t &modelHeight,const uint32_t &popDecodeFrameWaitTime,
                                                               const MultiChannelVideoReasoner* multiChannelVideoReasoner)
{
    // set device
    MxBase::DeviceContext device;
    device.devId = multiChannelVideoReasoner->deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->SetDevice(device);
    if (ret != APP_ERR_OK) {
        LogError << "SetDevice failed";
        return;
    }

    auto imageResizer = multiChannelVideoReasoner->imageResizer;
    auto resnetDetector = multiChannelVideoReasoner->resnetDetector;
    auto decodeFrameQueueMap = multiChannelVideoReasoner->decodeFrameQueueMap;
    auto videoFrameInfos = multiChannelVideoReasoner->videoFrameInfos;
    auto frameSkippingSampling = multiChannelVideoReasoner->frameSkippingSampling;
    while(true) {
        if (multiChannelVideoReasoner->stopFlag) {
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
        for ( iter = decodeFrameQueueMap.begin(); iter != decodeFrameQueueMap.end(); iter++) {

            auto rtspIndex = iter->first;
            auto decodeFrameQueue = iter->second;
            if (decodeFrameQueue->IsEmpty()) {
                continue;
            }

            ret = frameSkippingSampling->Process();
            if (ret != APP_ERR_OK) {
                LogError << "FrameSkippingSampling failed";
                continue;
            }

            // get decode frame data
            std::shared_ptr<void> data = nullptr;
            ret = decodeFrameQueue->Pop(data, popDecodeFrameWaitTime);
            if (ret != APP_ERR_OK) {
                LogError << "Pop failed";
                continue;
            }
            auto decodeFrame = std::make_shared<MxBase::MemoryData>();
            decodeFrame = std::static_pointer_cast<MxBase::MemoryData>(data);

            if(frameSkippingSampling->stopFlag) {
                LogError << "resize frame and iter :";
                // resize frame
                MxBase::DvppDataInfo resizeFrame = {};
                ret = imageResizer->ResizeFromMemory(*decodeFrame, videoFrameInfos[rtspIndex].width, videoFrameInfos[rtspIndex].height,
                                                     modelWidth, modelHeight, resizeFrame);
                if (ret != APP_ERR_OK) {
                    LogError << "Resize image failed, ret = " << ret << " " << GetError(ret);
                    continue;
                }

                // resnet detect
                std::vector<std::vector<MxBase::ClassInfo>> objInfos;
                ret = resnetDetector->Detect(resizeFrame, objInfos, videoFrameInfos[rtspIndex].width, videoFrameInfos[rtspIndex].height);
                if (ret != APP_ERR_OK) {
                    LogError << "Resnet detect image failed, ret = " << ret << " " << GetError(ret) << ".";
                    continue;
                }

                // save detect result
                ret = Util::SaveResult(decodeFrame, resizeFrame.frameId, objInfos,
                                       videoFrameInfos[rtspIndex].width, videoFrameInfos[rtspIndex].height, rtspIndex);
            }

            if (ret != APP_ERR_OK) {
                LogError << "Save result failed, ret=" << ret << ".";
                return;
            }
        }
    }
}

/// ========== private Method ========== ///
APP_ERROR MultiChannelVideoReasoner::CreateStreamPullerAndVideoDecoder(const ReasonerConfig &config)
{
    auto rtspList = config.rtspList;

    APP_ERROR ret;
    AscendStreamPuller::VideoFrameInfo videoFrameInfo;
    AscendVideoDecoder::DecoderInitParam decoderInitParam = {};

    for (uint32_t i = 0; i < rtspList.size(); i++) {
        auto streamPuller = std::make_shared<AscendStreamPuller::StreamPuller>();
        auto videoDecoder = std::make_shared<AscendVideoDecoder::VideoDecoder>();

        ret = streamPuller->Init(rtspList[i], config.maxTryOpenVideoStream, config.deviceId);
        if (ret != APP_ERR_OK) {
            LogError << "Init " << i + 1 << " StreamPuller failed, stream name: " << rtspList[i];
            return ret;
        }
        videoFrameInfo = streamPuller->GetFrameInfo();

        Util::InitVideoDecoderParam(decoderInitParam, config.deviceId, config.baseVideoChannelId + i, videoFrameInfo);
        ret = videoDecoder->Init(decoderInitParam);
        if (ret != APP_ERR_OK) {
            LogError << "Init " << i + 1 << " VideoDecoder failed";
            return ret;
        }

        // save
        streamPullers.push_back(streamPuller);
        videoDecoders.push_back(videoDecoder);
        videoFrameInfos.push_back(videoFrameInfo);
    }

    return APP_ERR_OK;
}
APP_ERROR MultiChannelVideoReasoner::CreateFrameSkippingSampling(const ReasonerConfig &config) {
    APP_ERROR ret;
    frameSkippingSampling = std::make_shared<AscendFrameSkippingSampling::FrameSkippingSampling>();
    ret = frameSkippingSampling->Init(config.maxSamplingInterval, config.SamplingInterval, config.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Init SamplingInterval failed";
        return ret;
    }
    return APP_ERR_OK;
}


APP_ERROR MultiChannelVideoReasoner::CreateImageResizer(const ReasonerConfig &config)
{
    imageResizer = std::make_shared<AscendImageResizer::ImageResizer>();

    APP_ERROR ret = imageResizer->Init(config.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Init image resizer failed";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR MultiChannelVideoReasoner::CreateResnetDetector(const ReasonerConfig &config)
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

APP_ERROR MultiChannelVideoReasoner::DestroyStreamPullerAndVideoDecoder()
{
    APP_ERROR ret;
    // deinit stream pullers and video decoders
    for (uint32_t i = 0; i < streamPullers.size(); i++) {
        // deinit video decoder
        ret = videoDecoders[i]->DeInit();
        if (ret != APP_ERR_OK) {
            LogError << "Deinit " << i + 1 << " VideoDecoder failed";
            return ret;
        }

        // deinit stream puller
        ret = streamPullers[i]->DeInit();
        if (ret != APP_ERR_OK) {
            LogError << "Deinit " << i + 1 << " StreamPuller failed.";
            return ret;
        }
    }

    return APP_ERR_OK;
}

APP_ERROR MultiChannelVideoReasoner::DestroyFrameSkippingSampling() {
    APP_ERROR ret = frameSkippingSampling->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "FrameSkippingSampling DeInit failed.";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR MultiChannelVideoReasoner::DestroyImageResizer()
{
    APP_ERROR ret = imageResizer->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "ImageResizer DeInit failed.";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR MultiChannelVideoReasoner::DestroyResnetDetector()
{
    APP_ERROR ret = resnetDetector->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "ResnetDetector DeInit failed.";
        return ret;
    }
    return APP_ERR_OK;
}

void MultiChannelVideoReasoner::ClearData() {
    // stop and clear queue
    std::_Rb_tree_const_iterator<std::pair<const int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>>> iter;
    for ( iter = decodeFrameQueueMap.begin(); iter != decodeFrameQueueMap.end(); iter++) {
        iter->second->Stop();
        iter->second->Clear();
    }
    decodeFrameQueueMap.clear();
    videoFrameInfos.clear();
    videoDecoders.clear();
    streamPullers.clear();
}

