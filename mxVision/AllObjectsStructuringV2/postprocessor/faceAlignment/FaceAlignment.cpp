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

#include "opencv2/opencv.hpp"
#include "FaceAlignment.h"

void FaceAlignment::DestoryMemory(std::vector<MxBase::DvppDataInfo> &outputDataInfoVec)
{
    for (size_t i = 0; i < outputDataInfoVec.size(); i++)
    {
        if (outputDataInfoVec[i].data != nullptr && outputDataInfoVec[i].destory != nullptr)
        {
            outputDataInfoVec[i].destory(outputDataInfoVec[i].data);
        }
    }
}

APP_ERROR FaceAlignment::Process(std::vector<MxBase::Image> &intputImageVec, std::vector<MxBase::Image> &outputImageVec,
                                 std::vector<MxBase::KeyPointInfo> &KeyPointInfoVec, int picHeight, int picWidth, int deviceID = 0)
{
    std::vector<MxBase::DvppDataInfo> inputDataInfoVec;
    std::vector<MxBase::DvppDataInfo> outputDataInfoVec;

    // Image --> Dvpp
    for (size_t i = 0; i < intputImageVec.size(); i++)
    {
        MxBase::DvppDataInfo inputDataInfo;
        MxBase::Image intputImage = intputImageVec[i];
        int imageDeviceId = intputImage.GetDeviceId();
        if (imageDeviceId != deviceID)
        {
            inputDataInfoVec.clear();
            LogError << "image deviceId != input deviceID , no image will be processed !";
            break;
        }
        intputImage.ToHost();
        MxBase::Size inputImageSize = intputImage.GetSize();
        MxBase::Size inputImageOriSize = intputImage.GetOriginalSize();
        inputDataInfo.width = inputImageOriSize.width;
        inputDataInfo.height = inputImageOriSize.height;
        inputDataInfo.widthStride = inputImageSize.width;
        inputDataInfo.heightStride = inputImageSize.height;
        inputDataInfo.dataSize = intputImage.GetDataSize();
        inputDataInfo.data = intputImage.GetData();
        inputDataInfoVec.push_back(inputDataInfo)
    }
    if (inputDataInfoVec.empty())
    {
        LogError << "inputDataInfoVec for warp_affine empty!";
        DestoryMemory(outputDataInfoVec);
        return APP_ERR_INVALID_PARAM;
    }

    // do warpAffine
    outputDataInfoVec.resize(inputDataInfoVec.size());
    APP_ERROR ret = warpAffine_.Process(inputDataInfoVec, outputDataInfoVec, KeyPointInfoVec, picHeight, picWidth);
    if (ret != APP_ERR_OK)
    {
        LogError << "Face warp affine failed!";
        DestoryMemory(outputDataInfoVec);
        return ret;
    }

    // Dvpp --> Image
    for (size_t i = 0; i < outputDataInfoVec.size(); i++)
    {
        MxBase::DvppDataInfo outputDataInfo = outputDataInfoVec[i];
        MxBase::Size outImageSize(outputDataInfoVec.widthStride, outputDataInfoVec.heightStride);

        MxBase::MemoryaData srcData(static_cast<void *>(outputDataInfo.data), outputDataInfo.dataSize, MxBase::MemoryaData::MEMORY_DEVICE);
        MxBase::MemoryaData dstData(outputDataInfo.dataSize, MxBase::MemoryaData::MEMORY_HOST_MALLOC);
        APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(dstData, srcData);
        if (ret != APP_ERR_OK)
        {
            LogError << "MxbsMallocAndCopy failed!";
            DestoryMemory(outputDataInfoVec);
            return ret;
        }
        MxBase::MemoryHelper::MxbsFree(srcData);
        MxBase::Image outputImage(static_cast<shared_ptr<uint8_t *>>(uint8_t * dstData.ptrData), dstData.size, deviceID, outImageSize,
                                  static_cast<MxBase::ImageFormat>(outputDataInfo.format));
        outputImageVec.push_back(outputImage);
    }

    return APP_ERR_OK;
}
