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

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/Log/Log.h"
#include "opencv2/opencv.hpp"

#include <dirent.h>
#include <cstdlib>

#include "../StreamPuller/StreamPuller.h"
#include "../VideoDecoder/VideoDecoder.h"

namespace {
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
}
class Util {

public:
    static void InitVideoDecoderParam(AscendVideoDecoder::DecoderInitParam &initParam, const uint32_t deviceId, const uint32_t channelId,
                                      const AscendStreamPuller::VideoFrameInfo &videoFrameInfo)
    {
        initParam.deviceId = deviceId;
        initParam.channelId = channelId;
        initParam.inputVideoFormat = videoFrameInfo.format;
        initParam.inputVideoHeight = videoFrameInfo.height;
        initParam.inputVideoWidth = videoFrameInfo.width;
        initParam.outputImageFormat = MxBase::MXBASE_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    }

    static void InitResnetParam(AscendResnetDetector::ResnetInitParam &initParam, const uint32_t deviceId, const std::string &labelPath,
                              const std::string &modelPath)
    {
        initParam.deviceId = deviceId;
        initParam.labelPath = labelPath;
        initParam.checkTensor = true;
        initParam.modelPath = modelPath;
        initParam.classNum = 21;
        initParam.biasesNum = 18;
        initParam.biases = "10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326";
        initParam.objectnessThresh = "0.001";
        initParam.iouThresh = "0.5";
        initParam.scoreThresh = "0.001";
        initParam.resnetType = 3;
        initParam.modelType = 0;
        initParam.inputType = 0;
        initParam.anchorDim = 3;
    }

    static bool IsExistDataInQueueMap( const std::map<int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> &decodeFrameQueueMap)
    {
        std::_Rb_tree_const_iterator<std::pair<const int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>>> iter;
        for ( iter = decodeFrameQueueMap.begin(); iter != decodeFrameQueueMap.end(); iter++) {
            if (!iter->second->IsEmpty()) {
                return true;
            }
        }

        return false;
    }

    static APP_ERROR SaveResult(std::shared_ptr<MxBase::MemoryData> resultInfo, const uint32_t frameId,
                                const std::vector<std::vector<MxBase::ClassInfo>> objInfos,
                                const uint32_t videoWidth, const uint32_t videoHeight,
                                const int rtspIndex = 1)
                                {
        MxBase::MemoryData memoryDst(resultInfo->size,MxBase::MemoryData::MEMORY_HOST_NEW);
        APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst, *resultInfo);
        if(ret != APP_ERR_OK){
            LogError << "Fail to malloc and copy host memory.";
            return ret;
        }
        cv::Mat imgYuv = cv::Mat(videoHeight * YUV_BYTE_NU / YUV_BYTE_DE, videoWidth, CV_8UC1, memoryDst.ptrData);
        cv::Mat imgBgr = cv::Mat(videoHeight, videoWidth, CV_8UC3);
        cv::cvtColor(imgYuv, imgBgr, cv::COLOR_YUV2BGR_NV12);

        std::vector<MxBase::ClassInfo> info;
        for (uint32_t i = 0; i < objInfos.size(); i++) {
            float maxConfidence = 0;
            uint32_t index;
            for (uint32_t j = 0; j < objInfos[i].size(); j++) {
                LogDebug << objInfos[i][j].confidence;
                if (objInfos[i][j].confidence > maxConfidence) {
                    maxConfidence = objInfos[i][j].confidence;
                    index = j;
                }
            }
            info.push_back(objInfos[i][index]);
//            LogInfo << "id: " << info[i].classId << "; lable: " << info[i].className
//            << "; confidence: " << info[i].confidence
//            << "; box: [ (" << info[i].x0 << "," << info[i].y0 << ") "
//            << "(" << info[i].x1 << "," << info[i].y1 << ") ]";

            const cv::Scalar green = cv::Scalar(255, 0, 0);
            const uint32_t thickness = 4;
//            const uint32_t xOffset = 10;
//            const uint32_t yOffset = 10;
            const uint32_t lineType = 8;
            const float fontScale = 2.0;

            cv::putText(imgBgr, info[i].className, cv::Point(100, videoWidth / 2),//cv::Point(info[i].x0 + xOffset, info[i].y0 + yOffset),
                        cv::FONT_HERSHEY_SIMPLEX, fontScale, green, thickness, lineType);
//            cv::rectangle(imgBgr,cv::Rect(info[i].x0, info[i].y0,
//                                          info[i].x1 - info[i].x0, info[i].y1 - info[i].y0),
//                          green, thickness);

            // write result
            std::string resultDir = "./result";
            if (opendir(resultDir.c_str()) == NULL) {
                LogDebug << "result dir not exist. create it!";
                std::string command = "mkdir -p " + resultDir;
                system(command.c_str());
            }
            resultDir = resultDir + "/rtsp" + std::to_string(rtspIndex);
            if (opendir(resultDir.c_str()) == NULL) {
                LogDebug << "result dir not exist. create it!";
                std::string command = "mkdir -p " + resultDir;
                system(command.c_str());
            }

            std::string fileName = resultDir + "/" + std::to_string(frameId + 1) + ".jpg";
            cv::imwrite(fileName, imgBgr);
        }

        ret = MxBase::MemoryHelper::MxbsFree(memoryDst);
        if(ret != APP_ERR_OK){
            LogError << "Fail to MxbsFree memory.";
            return ret;
        }
        return APP_ERR_OK;
    }
};


#endif //MULTICHANNELVIDEODETECTION_UTIL_H
