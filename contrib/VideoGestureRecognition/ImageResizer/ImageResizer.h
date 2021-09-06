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

#ifndef VIDEOGESTUREREASONER_IMAGERESIZER_H
#define VIDEOGESTUREREASONER_IMAGERESIZER_H

#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "../BlockingQueue/BlockingQueue.h"

namespace AscendImageResizer {
    struct ImageResizerParma{
        uint32_t originWidth;
        uint32_t originHeight;
        uint32_t resizeWidth;
        uint32_t resizeHeight;
    };
class ImageResizer {
public:
    ImageResizer() = default;
    ~ImageResizer() = default;

    APP_ERROR Init(uint32_t deviceId);
    APP_ERROR DeInit();
    APP_ERROR Resize(MxBase::DvppDataInfo &inputImageInfo,
                     const uint32_t &resizeWidth, const uint32_t &resizeHeight,
                     MxBase::DvppDataInfo &outputImageInfo);
    APP_ERROR ResizeFromMemory(MxBase::MemoryData &imageInfo, const ImageResizerParma &imageInitParma,
                     MxBase::DvppDataInfo &outputImageInfo);

public:
    // running flag
    bool stopFlag;

private:
    // image processor
    std::shared_ptr<MxBase::DvppWrapper> vDvppWrapper;

    // device id
    uint32_t deviceId;
};
} // end AscendImageResizer
#endif // MULTICHANNELVIDEODETECTION_IMAGERESIZER_H
