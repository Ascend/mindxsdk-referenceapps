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

#ifndef MULTICHANNELVIDEODETECTION_UTIL_H
#define MULTICHANNELVIDEODETECTION_UTIL_H

#include <dirent.h>
#include <cstdlib>
#include "MxBase/Log/Log.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "opencv2/opencv.hpp"
#include "../StreamPuller/StreamPuller.h"
#include "../VideoDecoder/VideoDecoder.h"
#include "../BlockingQueue/BlockingQueue.h"
#include "../ResnetDetector/ResnetDetector.h"

class Util {
public:
    Util() = default;
    ~Util() = default;

    static void InitVideoDecoderParam(AscendVideoDecoder::DecoderInitParam &initParam,
                                      const uint32_t deviceId, const uint32_t channelId,
                                      const AscendStreamPuller::VideoFrameInfo &videoFrameInfo);

    static void InitResnetParam(AscendResnetDetector::ResnetInitParam &initParam,
                                const uint32_t deviceId, const std::string &labelPath,
                                const std::string &modelPath);

    static bool IsExistDataInQueueMap(const std::map<int,
                                       std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> &decodeFrameQueueMap);

    static APP_ERROR SaveResult(std::shared_ptr<MxBase::MemoryData> resultInfo, const uint32_t frameId,
                                const std::vector<std::vector<MxBase::ClassInfo>>& objInfos,
                                const uint32_t videoWidth, const uint32_t videoHeight,
                                const int rtspIndex);

private:
    // class num
    static const uint32_t CLASS_LABEL_NUM = 21;
    // biases num
    static const uint32_t BIASES_LABEL_NUM = 18;
    // resnet type
    static const uint32_t RESNET_TYPE = 3;
    // anchor dim
    static const uint32_t ANCHOR_DIM = 3;
    // yuv type
    static const uint32_t YUV_BYTE_NU = 3;
    static const uint32_t YUV_BYTE_DE = 2;
    static const uint32_t POINT_TYPE = 2;
};

#endif // MULTICHANNELVIDEODETECTION_UTIL_H
