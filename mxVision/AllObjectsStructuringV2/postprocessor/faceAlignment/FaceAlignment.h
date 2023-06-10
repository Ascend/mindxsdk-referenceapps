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

#ifndef FACEALIGNMENT_FACEALIGNMENT_H
#define FACEALIGNMENT_FACEALIGNMENT_H
#include <iostream>
#include <vector>
#include <algorithm>
#include "MxBase/CV/WarpAffine/WarpAffine.h" // MxBase::KeyPointInfo & MxBase::WarpAffinr
#include "MxBase/MxBase.h"
#include "MxBase/E2eInfer/Image/Image.h"    // V2 Image
#include "MxBase/E2eInfer/Size/Size.h"      // V2 Size
#include "MxBase/ErrorCode/ErrorCode.h"     // APP_ERROR & APP_ERR_OK
#include "MxBase/DvppWrapper/DvppWrapper.h" // MxBase::DvppDataInfo

class FaceAlignment
{
public:
    APP_ERROR Process(std::vector<MxBase::Image> &intputImageVec, std::vector<MxBase::Image> &outputImageVec,
                      std::vector<MxBase::KeyPointInfo> &KeyPointInfoVec, int picHeight, int picWidth, int deviceID) private : void DestoryMemory(std::vector<MxBase::DvppDataInfo> &outputDataInfoVec);

    MxBase::WarpAffinr warpAffinr_;
};

#endif /* FACEALIGNMENT_FACEALIGNMENT_H */
