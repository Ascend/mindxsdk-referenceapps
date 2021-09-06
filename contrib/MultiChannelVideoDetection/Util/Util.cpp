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

/**
 * Init {@link VideoDecoder} initial param
 * @param initParam reference to {@link DecoderInitParam}
 * @param deviceId device id
 * @param channelId channel id
 * @param videoFrameInfo const reference to {@link VideoFrameInfo}
 */
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

/**
 * Init {@link YoloDetector} initial param
 * @param initParam reference to {@link YoloInitParam}
 * @param deviceId device id
 * @param labelPath const reference to yolo label path
 * @param modelPath const reference to yolo model path
 */
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

/**
 * Judge whether exist data in all queues
 * @param queueMap const reference to queue map
 * @return whether exist data in all queues
 */
bool Util::IsExistDataInQueueMap
        (const std::map<int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> &queueMap)
{
    std::_Rb_tree_const_iterator<std::pair<const int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>>> iter;
    for (iter = queueMap.begin(); iter != queueMap.end(); iter++) {
        if (!iter->second->IsEmpty()) {
            return true;
        }
    }

    return false;
}

/**
 * Stop and clear queue map
 * @param queueMap const reference to queue map
 */
void Util::StopAndClearQueueMap
        (const std::map<int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> &queueMap)
{
    std::_Rb_tree_const_iterator<std::pair<const int, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>>> iter;
    for (iter = queueMap.begin(); iter != queueMap.end(); iter++) {
        if (!iter->second->IsStop()) {
            iter->second->Stop();
            iter->second->Clear();
            LogInfo << "stop " << iter->first << " queue.";
        }
    }
}

/**
 * Get yolo detect result from objInfos
 * >> strategy: choose max confidence as final detect result
 *
 * @param objInfos const reference to the list of objectInfo list
 * @param rtspIndex curr rtsp stream index (used to print detect message)
 * @param frameId curr video frame id (used to print detect message)
 * @param printResult whether print detect result
 * @return yolo detect results
 */
std::vector<MxBase::ObjectInfo> Util::GetDetectionResult
        (const std::vector<std::vector<MxBase::ObjectInfo>> &objInfos,
         uint32_t rtspIndex, uint32_t frameId, bool printResult)
{
    std::vector<MxBase::ObjectInfo> info;

    // get max confidence as detection result
    for (uint32_t i = 0; i < objInfos.size(); i++) {
        float maxConfidence = 0;
        int32_t index = -1;
        for (uint32_t j = 0; j < objInfos[i].size(); j++) {
            if (objInfos[i][j].confidence > maxConfidence) {
                maxConfidence = objInfos[i][j].confidence;
                index = (int32_t)j;
            }
        }

        if (index == -1) {
            continue;
        }

        info.push_back(objInfos[i][index]);
        if (printResult) {
            LogInfo << "rtsp " << rtspIndex << " frame " << frameId
            << " result:{id: " << info[i].classId
            << "; label: " << info[i].className
            << "; confidence: " << info[i].confidence
            << "; box: [(" << info[i].x0 << "," << info[i].y0 << ")"
            << "(" << info[i].x1 << "," << info[i].y1 << ")]}.";
        }
    }

    return info;
}

/**
 * Check whether exist result dir, if not, create it!
 * @param totalVideoStreamNum number of total video streams
 */
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

/**
 * Save each channel video detect result to different dir
 * @param videoFrame const reference to the memory data of curr video frame
 * @param results const reference to the detect results (used to draw result on pic)
 * @param videoFrameInfo const reference to curr channel video frame info
 * @param frameId curr video frame id (used as save file name)
 * @param rtspIndex curr video stream index (used to choose save dir)
 * @return status code of whether saving result is successful
 */
APP_ERROR Util::SaveResult(const std::shared_ptr<MxBase::MemoryData> &videoFrame,
                           const std::vector<MxBase::ObjectInfo> &results,
                           const AscendStreamPuller::VideoFrameInfo &videoFrameInfo,
                           uint32_t frameId, uint32_t rtspIndex)
{
    MxBase::MemoryData memoryDst(videoFrame->size, MxBase::MemoryData::MEMORY_HOST_NEW);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst, *videoFrame);
    if (ret != APP_ERR_OK) {
        LogError << "Fail to malloc and copy host memory.";
        return ret;
    }

    // get video frame origin size
    auto videoHeight = videoFrameInfo.height;
    auto videoWidth = videoFrameInfo.width;

    // origin decode yuv image of video frame
    int32_t yuvHeight = (int32_t)(videoHeight * AscendYoloDetector::YUV_BYTE_NU / AscendYoloDetector::YUV_BYTE_DE);
    cv::Mat imgYuv = cv::Mat(yuvHeight, (int32_t)videoWidth, CV_8UC1, memoryDst.ptrData);
    // save result rgb image
    cv::Mat imgBgr = cv::Mat((int32_t)videoHeight, (int32_t)videoWidth, CV_8UC3);

    // converse color space from imgYuv to imgBgr
    cv::cvtColor(imgYuv, imgBgr, cv::COLOR_YUV2BGR_NV12);

    for (const auto & result : results) {
        // set text and rectangle color as green
        const cv::Scalar green = cv::Scalar(0, 255, 0);
        // set text thickness as 4
        const uint32_t thickness = 4;
        // set text-draw x-offset as 10
        const uint32_t xOffset = 10;
        // set text-draw y-offset as 10
        const uint32_t yOffset = 10;
        // set text line type as 8-connected type
        const uint32_t lineType = 8;
        // set text font scale as 1.0 (relative to font base size)
        const float fontScale = 1.0;

        // draw detect result on result pic
        cv::putText(imgBgr, result.className,
                    cv::Point((int)(result.x0 + xOffset), (int)(result.y0 + yOffset)),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, green, thickness, lineType);
        // draw edge of detect result object on result pic
        cv::rectangle(imgBgr,
                      cv::Rect((int)result.x0, (int)result.y0,
                               (int)(result.x1 - result.x0), (int)(result.y1 - result.y0)),
                      green, thickness);

        // write result as (frameId + 1).jpg
        std::string resultDir = "./result/rtsp" + std::to_string(rtspIndex);
        std::string fileName = resultDir + "/" + std::to_string(frameId + 1) + ".jpg";
        cv::imwrite(fileName, imgBgr);
    }

    ret = MxBase::MemoryHelper::MxbsFree(memoryDst);
    if (ret != APP_ERR_OK) {
        LogError << "Fail to MxbsFree memory.";
        return ret;
    }
    return APP_ERR_OK;
}

/// ===== private method ===== ///

/**
 * Create dir
 * @param path const reference to dir path
 */
void Util::CreateDir(const std::string &path)
{
    LogInfo << path << " not exist. create it!";
    std::string command = "mkdir -p " + path;

    const int32_t commonFailedCode = -1;
    const int32_t normalTerminationCode = 0;

    int32_t code = system(command.c_str());
    if (code != commonFailedCode && WIFEXITED(code) && WEXITSTATUS(code) == normalTerminationCode) {
        LogInfo << "create " << path << " successfully.";
    } else {
        LogError << "create " << path << " failed, code = " << WEXITSTATUS(code) << ".";
    }
}