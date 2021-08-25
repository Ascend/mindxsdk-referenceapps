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

#include "VideoDecoder.h"
#include "BlockingQueue/BlockingQueue.h"

#include "MxBase/Log/Log.h"

namespace AscendVideoDecoder {

/**
 * Init VideoDecoder
 * @param initParam const reference to initial param
 * @return status code of whether initialization is successful
 */
APP_ERROR VideoDecoder::Init(const DecoderInitParam &initParam)
{
    LogInfo << "VideoDecoder init start.";

    this->channelId = initParam.channelId;
    this->deviceId = initParam.deviceId;
    this->frameWidth = initParam.inputVideoWidth;
    this->frameHeight = initParam.inputVideoHeight;

    APP_ERROR ret = InitDvppWrapper(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Init DvppWrapper failed.";
        return ret;
    }

    LogInfo << "VideoDecoder init successful.";
    return APP_ERR_OK;
}

/**
 * De-init VideoDecoder
 * @return status code of whether de-initialization is successful
 */
APP_ERROR VideoDecoder::DeInit()
{
    LogInfo << "VideoDecoder deinit start.";

    APP_ERROR ret = vDvppWrapper->DeInitVdec();
    if (ret != APP_ERR_OK) {
        LogError << "Failed to deinit dvppWrapper.";
        return ret;
    }

    LogInfo << "VideoDecoder deinit successful.";
    return APP_ERR_OK;
}

/**
 * Decode video frame memory data to specific format image
 * @param streamData reference to curr video frame memory data
 * @param userData pointer to user data (ex: decode frame queue)
 * @return status code of whether decode is successful
 */
APP_ERROR VideoDecoder::Decode(MxBase::MemoryData &streamData, void *userData)
{
    MxBase::MemoryData dvppMemory((size_t)streamData.size, MxBase::MemoryData::MEMORY_DVPP, this->deviceId);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(dvppMemory, streamData);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to MxbsMallocAndCopy.";
        return ret;
    }
    MxBase::DvppDataInfo inputDataInfo;
    inputDataInfo.dataSize = dvppMemory.size;
    inputDataInfo.data = (uint8_t *)dvppMemory.ptrData;
    inputDataInfo.height = this->frameHeight;
    inputDataInfo.width = this->frameWidth;
    inputDataInfo.channelId = this->channelId;
    inputDataInfo.frameId = frameId;

    // call DvppWrapper function to complete video frame data decode
    ret = vDvppWrapper->DvppVdec(inputDataInfo, userData);
    if (ret != APP_ERR_OK) {
        LogError << "DvppVdec Failed on frame " << frameId;
        MxBase::MemoryHelper::MxbsFree(dvppMemory);
        frameId++;
        return ret;
    }

    frameId++;
    return APP_ERR_OK;
}

/**
 * Get number of total decoded video frame
 * @return number of total decoded video frame
 */
uint32_t VideoDecoder::GetTotalDecodeFrameNum() const
{
    return frameId;
}

/// ========== private Method ========== ///

/**
 * Init DvppWrapper
 * @param initParam const reference to initial param {@link DecoderInitParam}
 * @return status code of whether DvppWrapper initialization is successful
 */
APP_ERROR VideoDecoder::InitDvppWrapper(const DecoderInitParam &initParam)
{
    // init decode config
    MxBase::VdecConfig vdecConfig;
    vdecConfig.inputVideoFormat = initParam.inputVideoFormat;
    vdecConfig.outputImageFormat = initParam.outputImageFormat;
    vdecConfig.deviceId = initParam.deviceId;
    vdecConfig.channelId = initParam.channelId;
    vdecConfig.callbackFunc = VideoDecodeCallback;
    vdecConfig.outMode = 1;

    vDvppWrapper = std::make_shared<MxBase::DvppWrapper>();
    if (vDvppWrapper == nullptr) {
        LogError << "Failed to create DvppWrapper.";
        return APP_ERR_COMM_INIT_FAIL;
    }

    APP_ERROR ret = vDvppWrapper->InitVdec(vdecConfig);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to init DvppWrapper.";
        return ret;
    }

    return APP_ERR_OK;
}

/**
 * >> static method
 * Callback of DvppWrapper decode video frame
 * @param buffer
 * @param inputDataInfo reference to the user input video frame data
 * @param userData pointer of user data
 * @return status code of whether callback is normal
 */
APP_ERROR VideoDecoder::VideoDecodeCallback(std::shared_ptr<void> buffer,
                                            MxBase::DvppDataInfo &inputDataInfo, void *userData)
{
    LogDebug << "decode frame " << inputDataInfo.frameId << " complete.";

    auto deleter = [] (MxBase::MemoryData* memoryData) {
        if (memoryData == nullptr) {
            LogError << "MxbsFree failed.";
            return;
        }

        APP_ERROR ret = MxBase::MemoryHelper::MxbsFree(*memoryData);
        delete memoryData;
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << " MxbsFree failed.";
            return;
        }
    };

    auto output = std::shared_ptr<MxBase::MemoryData>(
            new MxBase::MemoryData(buffer.get(),(size_t) inputDataInfo.dataSize,
                                   MxBase::MemoryData::MEMORY_DVPP, inputDataInfo.frameId), deleter);

    if (userData == nullptr) {
        LogInfo << "userData is null.";
        return APP_ERR_COMM_INVALID_POINTER;
    }

    // put decoded frame into decoded frame queue
    auto* decodeFrameQueue = (BlockingQueue<std::shared_ptr<void>>*) userData;
    decodeFrameQueue->Push(output, true);
    return APP_ERR_OK;
}

} // end AscendVideoDecoder