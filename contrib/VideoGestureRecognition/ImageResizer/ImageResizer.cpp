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

#include "ImageResizer.h"
#include "MxBase/Log/Log.h"

namespace AscendImageResizer {

APP_ERROR ImageResizer::Init(uint32_t deviceId)
{
    LogDebug << "ImageResizer init start.";
    this->deviceId = deviceId;

    // init DvppWrapper
    vDvppWrapper = std::make_shared<MxBase::DvppWrapper>();
    APP_ERROR ret = vDvppWrapper->Init();
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper init failed, ret = " << ret << ".";
        return ret;
    }

    stopFlag = false;
    LogDebug << "ImageResizer init successful.";
    return  APP_ERR_OK;
}

APP_ERROR ImageResizer::DeInit()
{
    LogDebug << "ImageResizer deinit start.";

    APP_ERROR ret = vDvppWrapper->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper deinit failed.";
        return ret;
    }

    stopFlag = true;

    LogDebug << "ImageResizer deinit successful.";
    return APP_ERR_OK;
}

APP_ERROR ImageResizer::Resize(MxBase::DvppDataInfo &inputImageInfo,
                               const uint32_t &resizeWidth, const uint32_t &resizeHeight,
                               MxBase::DvppDataInfo &outputImageInfo)
{
    // check image
    if (inputImageInfo.data == nullptr || inputImageInfo.dataSize <= 0 ||
        inputImageInfo.width <= 0 || inputImageInfo.height <= 0) {
        LogError << "Invalid image.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    MxBase::ResizeConfig resizeConfig = {};
    resizeConfig.width = resizeWidth;
    resizeConfig.height = resizeHeight;

    APP_ERROR ret = vDvppWrapper->VpcResize(inputImageInfo, outputImageInfo, resizeConfig);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "VpcResize failed.";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR ImageResizer::ResizeFromMemory(MxBase::MemoryData &imageInfo,
                                         const uint32_t &originWidth, const uint32_t &originHeight,
                                         const uint32_t &resizeWidth, const uint32_t &resizeHeight,
                                         MxBase::DvppDataInfo &outputImageInfo)
{
    // check image
    if (imageInfo.ptrData == nullptr || imageInfo.size <= 0) {
        LogError << "Invalid image.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    MxBase::DvppDataInfo input = {};
    input.width = originWidth;
    input.height = originHeight;
    input.widthStride = originWidth;
    input.heightStride = originHeight;
    input.dataSize = imageInfo.size;
    input.data = (uint8_t*) imageInfo.ptrData;
    input.frameId = imageInfo.deviceId;

    return Resize(input, resizeWidth, resizeHeight, outputImageInfo);
}

} // end AscendImageResizer
