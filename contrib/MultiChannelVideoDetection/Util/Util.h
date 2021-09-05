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

#include "StreamPuller/StreamPuller.h"
#include "VideoDecoder/VideoDecoder.h"
#include "YoloDetector/YoloDetector.h"
#include "BlockingQueue/BlockingQueue.h"

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/Log/Log.h"
#include "opencv2/opencv.hpp"

#include <dirent.h>
#include <cstdlib>

class Util {
public:
    static void InitVideoDecoderParam(AscendVideoDecoder::DecoderInitParam &initParam,
                                      uint32_t deviceId, uint32_t channelId,
                                      const AscendStreamPuller::VideoFrameInfo &videoFrameInfo);

    static void InitYoloParam(AscendYoloDetector::YoloInitParam &initParam, uint32_t deviceId,
                              const std::string &labelPath, const std::string &modelPath);

    static bool IsExistDataInQueueMap(
            const std::map<int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> &queueMap);

    static void StopAndClearQueueMap(
            const std::map<int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> &queueMap);

    static std::vector<MxBase::ObjectInfo> GetDetectionResult(
            const std::vector<std::vector<MxBase::ObjectInfo>> &objInfos,
            uint32_t rtspIndex, uint32_t frameId, bool printResult = true);

    static void CheckAndCreateResultDir(uint32_t totalVideoStreamNum);

    static APP_ERROR SaveResult(const std::shared_ptr<MxBase::MemoryData> &videoFrame,
                                const std::vector<MxBase::ObjectInfo> &results,
                                const AscendStreamPuller::VideoFrameInfo &videoFrameInfo,
                                uint32_t frameId, uint32_t rtspIndex = 0);

private:
    static void CreateDir(const std::string &path);

private:
    Util() = default;
    ~Util() = default;
};
#endif // MULTICHANNELVIDEODETECTION_UTIL_H
