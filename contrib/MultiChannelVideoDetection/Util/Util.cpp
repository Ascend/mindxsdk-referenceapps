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

#include "Util.h"

void Util::InitVideoDecoderParam(AscendVideoDecoder::DecoderInitParam &initParam,
                                 uint32_t deviceId, uint32_t channelId,
                                 const AscendStreamPuller::VideoFrameInfo &videoFrameInfo)
{
    initParam.deviceId = deviceId;
    initParam.channelId = channelId;
    initParam.inputVideoFormat = videoFrameInfo.format;
    initParam.inputVideoHeight = videoFrameInfo.height;
    initParam.inputVideoWidth = videoFrameInfo.width;
    initParam.outputImageFormat = MxBase::MXBASE_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
}

void Util::InitYoloParam(AscendYoloDetector::YoloInitParam &initParam, uint32_t deviceId,
                         const std::string &labelPath, const std::string &modelPath)
{
    initParam.deviceId = deviceId;
    initParam.labelPath = labelPath;
    initParam.checkTensor = true;
    initParam.modelPath = modelPath;
    initParam.classNum = AscendYoloDetector::YOLO_CLASS_NUM;
    initParam.biasesNum = AscendYoloDetector::YOLO_BIASES_NUM;
    initParam.biases = "10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326";
    initParam.objectnessThresh = "0.001";
    initParam.iouThresh = "0.5";
    initParam.scoreThresh = "0.001";
    initParam.yoloType = AscendYoloDetector::YOLO_TYPE;
    initParam.modelType = AscendYoloDetector::YOLO_MODEL_TYPE;
    initParam.inputType = AscendYoloDetector::YOLO_INPUT_TYPE;
    initParam.anchorDim = AscendYoloDetector::YOLO_ANCHOR_DIM;
}

bool Util::IsExistDataInQueueMap(const std::map<int,
                                 std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> &queueMap)
{
    std::_Rb_tree_const_iterator<std::pair<const int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>>> iter;
    for (iter = queueMap.begin();iter != queueMap.end();iter++) {
        if (!iter->second->IsEmpty()) {
            return true;
        }
    }

    return false;
}

std::vector<MxBase::ObjectInfo> Util::GetDetectionResult(
        const std::vector<std::vector<MxBase::ObjectInfo>>& objInfos,
        uint32_t rtspIndex, uint32_t frameId, bool printResult)
{
    std::vector<MxBase::ObjectInfo> info;

    // get max confidence as detection result
    for (uint32_t i = 0; i < objInfos.size(); i++) {
        float maxConfidence = 0;
        uint32_t index;
        for (uint32_t j = 0; j < objInfos[i].size(); j++) {
            if (objInfos[i][j].confidence > maxConfidence) {
                maxConfidence = objInfos[i][j].confidence;
                index = j;
            }
        }
        info.push_back(objInfos[i][index]);

        if (printResult) {
            LogInfo << "rtsp " << rtspIndex << " frame " << frameId
            <<" result:{id: " << info[i].classId
            << "; label: " << info[i].className
            << "; confidence: " << info[i].confidence
            << "; box: [(" << info[i].x0 << "," << info[i].y0 << ")"
            << "(" << info[i].x1 << "," << info[i].y1 << ")]}";
        }
    }

    return info;
}

void Util::CheckAndCreateResultDir(uint32_t totalVideoStreamNum)
{
    std::string resultDir = "./result";
    if (opendir(resultDir.c_str()) == nullptr) {
        CreateDir(resultDir);
    }

    for (uint32_t i = 0; i < totalVideoStreamNum; i++) {
        resultDir = "./result/rtsp" + std::to_string(i);
        if (opendir(resultDir.c_str()) == nullptr) {
            CreateDir(resultDir);
        }
    }
}

APP_ERROR Util::SaveResult(const std::shared_ptr<MxBase::MemoryData>& videoFrame,
                           const std::vector<MxBase::ObjectInfo>& results,
                           const AscendStreamPuller::VideoFrameInfo &videoFrameInfo,
                           uint32_t frameId, uint32_t rtspIndex)
{
    MxBase::MemoryData memoryDst(videoFrame->size, MxBase::MemoryData::MEMORY_HOST_NEW);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst, *videoFrame);
    if(ret != APP_ERR_OK){
        LogError << "Fail to malloc and copy host memory.";
        return ret;
    }

    // get video frame origin size
    auto videoHeight = videoFrameInfo.height;
    auto videoWidth = videoFrameInfo.width;

    cv::Mat imgYuv = cv::Mat((int32_t) (videoHeight *AscendYoloDetector::YUV_BYTE_NU /
            AscendYoloDetector::YUV_BYTE_DE), (int32_t) videoWidth, CV_8UC1, memoryDst.ptrData);
    cv::Mat imgBgr = cv::Mat((int32_t) videoHeight, (int32_t) videoWidth, CV_8UC3);
    cv::cvtColor(imgYuv, imgBgr, cv::COLOR_YUV2BGR_NV12);

    for (const auto & result : results) {
        const cv::Scalar green = cv::Scalar(0, 255, 0);
        const uint32_t thickness = 4;
        const uint32_t xOffset = 10;
        const uint32_t yOffset = 10;
        const uint32_t lineType = 8;
        const float fontScale = 1.0;

        cv::putText(imgBgr, result.className,
                    cv::Point((int) (result.x0 + xOffset), (int) (result.y0 + yOffset)),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, green, thickness, lineType);
        cv::rectangle(imgBgr,
                      cv::Rect((int) result.x0, (int) result.y0,
                               (int) (result.x1 - result.x0), (int) (result.y1 - result.y0)),
                      green, thickness);

        // write result as (frameId + 1).jpg
        std::string resultDir = "./result/rtsp" + std::to_string(rtspIndex);
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


///===== private method =====///
void Util::CreateDir(const std::string &path)
{
    LogInfo << path << " not exist. create it!";
    std::string command = "mkdir -p " + path;
    system(command.c_str());
}
 

