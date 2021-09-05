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
/**
 * Init ImageResizer
 * @param deviceId device id which main program use
 * @return status code of whether initialization is successful
 */
APP_ERROR ImageResizer::Init(uint32_t deviceId)
{
    LogInfo << "ImageResizer init start.";
    this->deviceId = deviceId;

    // init DvppWrapper
    vDvppWrapper = std::make_shared<MxBase::DvppWrapper>();
    APP_ERROR ret = vDvppWrapper->Init();
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper init failed, ret = " << ret << ".";
        return ret;
    }

    LogInfo << "ImageResizer init successful.";
    return  APP_ERR_OK;
}

/**
 * De-init ImageResizer
 * @return status code of whether de-initialization is successful
 */
APP_ERROR ImageResizer::DeInit()
{
    LogInfo << "ImageResizer deinit start.";

    APP_ERROR ret = vDvppWrapper->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper deinit failed.";
        return ret;
    }

    LogInfo << "ImageResizer deinit successful.";
    return APP_ERR_OK;
}

/**
 * Resize image with specific width and height
 * @param inputImageInfo reference to input image
 * @param resizeWidth width need to resize
 * @param resizeHeight height need to resize
 * @param outputImageInfo reference to output image
 * @return status code of whether resize image is successful
 */
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

    // construct resize config
    MxBase::ResizeConfig resizeConfig = {};
    resizeConfig.width = resizeWidth;
    resizeConfig.height = resizeHeight;

    // call DvppWrapper function to complete the image resize
    APP_ERROR ret = vDvppWrapper->VpcResize(inputImageInfo, outputImageInfo, resizeConfig);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "VpcResize failed.";
        return ret;
    }

    return APP_ERR_OK;
}

/**
 * Resize memory of image with specific width and height
 * @param imageInfo the memory data of input image
 * @param originWidth width of input image
 * @param originHeight height of input image
 * @param resizeWidth width need to resize
 * @param resizeHeight height need to resize
 * @param outputImageInfo reference to output image
 * @return status code of whether resize image is successful
 */
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

    // construct image by memory data
    MxBase::DvppDataInfo input = {};
    input.width = originWidth;
    input.height = originHeight;
    input.widthStride = originWidth;
    input.heightStride = originHeight;
    input.dataSize = imageInfo.size;
    input.data = (uint8_t*)imageInfo.ptrData;
    input.frameId = imageInfo.deviceId;

    return Resize(input, resizeWidth, resizeHeight, outputImageInfo);
}
} // end AscendImageResizer