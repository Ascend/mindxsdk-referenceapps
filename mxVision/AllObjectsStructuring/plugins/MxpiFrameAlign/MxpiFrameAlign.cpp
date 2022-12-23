/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "MxpiFrameAlign.h"
#include "MxBase/Log/Log.h"

using namespace MxBase;
using namespace MxTools;

namespace {
const int STREAM_DATA_INPUT_PORT_ID = 0;
const char SPLIT_RULE = ',';
const int ALIGN_NUM = 2;
}

APP_ERROR MxpiFrameAlign::Init(std::map<std::string, std::shared_ptr<void>> &configParamMap)
{
    LogInfo << "Begin to initialize MxpiFrameAlign(" << elementName_ << ").";
    if (status_ != ASYNC) {
        LogDebug << "element(" << elementName_
                 << ") status must be async(0), you set status sync(1), so force status to async(0).";
        status_ = ASYNC;
    }
    dataKeyVec_ = SplitWithRemoveBlank(dataSource_, SPLIT_RULE);
    if (dataKeyVec_.empty()) {
        LogError << GetError(APP_ERR_COMM_INIT_FAIL, elementName_)
                 << "the data source can not be null";
        return APP_ERR_COMM_INIT_FAIL;
    }
    intervalTime_ = *std::static_pointer_cast<int>(configParamMap["intervalTime"]);
    sendThread_ = std::thread(&MxpiFrameAlign::SendThread, this);

    LogInfo << "End to initialize MxpiFrameAlign(" << elementName_ << ").";
    return APP_ERR_OK;
}

APP_ERROR MxpiFrameAlign::DeInit()
{
    LogInfo << "Begin to deinitialize MxpiFrameAlign(" << elementName_ << ").";
    sendStop_ = true;
    if (sendThread_.joinable()) {
        sendThread_.join();
    }
    LogInfo << "End to deinitialize MxpiFrameAlign(" << elementName_ << ").";
    return APP_ERR_OK;
}

APP_ERROR MxpiFrameAlign::Process(std::vector<MxTools::MxpiBuffer*> &mxpiBuffer)
{
    LogDebug << "Begin to process MxpiFrameAlign(" << pluginName_ << ").";
    errorInfo_.str("");
    int inputPortId = -1;
    for (size_t i = 0; i < mxpiBuffer.size(); i++) {
        if (mxpiBuffer[i] != nullptr) {
            inputPortId = i;
            break;
        }
    }
    MxpiBuffer* inputBuffer = mxpiBuffer[inputPortId];
    MxTools::MxpiMetadataManager mxpiMetadataManager(*inputBuffer);
    if (mxpiMetadataManager.GetErrorInfo() != nullptr) {
        LogError << "Input data is invalid, element(" << elementName_ << ") plugin will not be executed rightly.";
        SendData(0, *inputBuffer);
        return APP_ERR_OK;
    }
    if (inputPortId == STREAM_DATA_INPUT_PORT_ID) {
        GetStreamData(*inputBuffer);
        MxpiBufferManager::DestroyBuffer(inputBuffer);
    } else {
        APP_ERROR ret = GetObjectList(*inputBuffer);
        if (ret != APP_ERR_OK) {
            LogError << errorInfo_.str();
            MxpiBufferManager::DestroyBuffer(inputBuffer);
            return ret;
        }
        AlignFrameObjectInfo();
        MxpiBufferManager::DestroyBuffer(inputBuffer);
    }

    LogDebug << "End to process MxpiFrameAlign(" << elementName_ << ").";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiFrameAlign::DefineProperties()
{
    std::vector<std::shared_ptr<void>> properties;
    auto intervalTime = std::make_shared<ElementProperty<int>>(ElementProperty<int> {
        INT, "intervalTime", "intervalTime", "the interval time of send data (ms)", 20, 0, 100
    });
    properties.push_back(intervalTime);
    return properties;
}

MxpiPortInfo MxpiFrameAlign::DefineInputPorts()
{
    MxpiPortInfo inputPortInfo;
    std::vector<std::vector<std::string>> value = {{"ANY"}};
    GenerateStaticInputPortsInfo(value, inputPortInfo);
    std::vector<std::vector<std::string>> featureCaps = {{"ANY"}};
    GenerateStaticInputPortsInfo(featureCaps, inputPortInfo);
    return inputPortInfo;
}

MxpiPortInfo MxpiFrameAlign::DefineOutputPorts()
{
    MxpiPortInfo outputPortInfo;
    std::vector<std::vector<std::string>> value = {{"ANY"}};
    GenerateStaticOutputPortsInfo(value, outputPortInfo);

    return outputPortInfo;
}

void MxpiFrameAlign::GetStreamData(MxTools::MxpiBuffer &inputBuffer)
{
    MxpiFrame mxpiFrame = MxpiBufferManager::GetHostDataInfo(inputBuffer);
    MxpiWebDisplayData webDisplayData {};
    uint32_t frameId = mxpiFrame.frameinfo().frameid();
    webDisplayData.set_channel_id(std::to_string(mxpiFrame.frameinfo().channelid()));
    webDisplayData.set_frame_index(frameId);
    webDisplayData.set_h264_size(mxpiFrame.visionlist().visionvec(0).visiondata().datasize());
    webDisplayData.set_h264_data((void*) mxpiFrame.visionlist().visionvec(0).visiondata().dataptr(),
                                 mxpiFrame.visionlist().visionvec(0).visiondata().datasize());
    StreamData streamData {};
    streamData.webDisplayData = webDisplayData;
    streamData.sendFlag = false;
    streamDataMap_.Insert(frameId, streamData);
}

APP_ERROR MxpiFrameAlign::GetObjectList(MxpiBuffer &inputBuffer)
{
    MxpiFrame frameData = MxTools::MxpiBufferManager::GetDeviceDataInfo(inputBuffer);
    uint32_t frameId = frameData.frameinfo().frameid();
    std::string channelId = std::to_string(frameData.frameinfo().channelid());
    MxpiMetadataManager mxpiMetadataManager(inputBuffer);
    if (!streamDataMap_.Find(frameId)) {
        return APP_ERR_OK;
    }
    std::vector<ObjectInfo> objectInfoList;
    for (size_t i = 0; i < dataKeyVec_.size(); ++i) {
        std::shared_ptr<MxpiTrackLetList> trackLetList = std::static_pointer_cast<MxpiTrackLetList>(
            mxpiMetadataManager.GetMetadata(dataKeyVec_[i]));
        if (trackLetList == nullptr || trackLetList->trackletvec_size() == 0) {
            continue;
        }
        // Get object detection result list by trackLet parent name from the buffer
        std::string parentName = "";
        for (uint32_t j = 0; j < trackLetList->trackletvec_size(); ++j) {
            if (trackLetList->trackletvec(j).headervec_size() > 0) {
                parentName = trackLetList->trackletvec(j).headervec(0).datasource();
                break;
            }
        }
        std::shared_ptr<MxpiObjectList> objectList = std::static_pointer_cast<MxpiObjectList>(
            mxpiMetadataManager.GetMetadata(parentName));
        if (objectList == nullptr || objectList->objectvec_size() == 0) {
            errorInfo_ << GetError(APP_ERR_COMM_INVALID_POINTER, pluginName_) << "Failed to get detection result.";
            continue;
        }
        for (int k = 0; k < trackLetList->trackletvec_size(); ++k) {
            if (trackLetList->trackletvec(k).headervec_size() == 0) {
                continue;
            }
            int32_t memberId = trackLetList->trackletvec(k).headervec(0).memberid();
            if (memberId >= objectList->objectvec_size()) {
                continue;
            }
            ObjectInfo objectInfo {};
            objectInfo.trackId = std::to_string(trackLetList->trackletvec(k).trackid()) + "_" +
                                 std::to_string(objectList->objectvec(memberId).classvec(0).classid());
            objectInfo.x0 = objectList->objectvec(memberId).x0();
            objectInfo.y0 = objectList->objectvec(memberId).y0();
            objectInfo.x1 = objectList->objectvec(memberId).x1();
            objectInfo.y1 = objectList->objectvec(memberId).y1();
            objectInfoList.push_back(objectInfo);
        }
    }
    objectListMap_.Insert(frameId, objectInfoList);
    return APP_ERR_OK;
}

bool MxpiFrameAlign::HadTrackId(std::vector<ObjectInfo> &objectInfoList, std::string &trackId)
{
    for (size_t i = 0; i < objectInfoList.size(); ++i) {
        if (trackId == objectInfoList[i].trackId) {
            return true;
        }
    }
    return false;
}

void MxpiFrameAlign::ObjectInfoInterpolated(std::vector<ObjectInfo> &interpolatedObjectInfoList,
                                            std::vector<ObjectInfo> &previousObjectInfoList,
                                            std::vector<ObjectInfo> &latterObjectInfoList,
                                            float &offset)
{
    for (size_t i = 0; i < latterObjectInfoList.size(); ++i) {
        std::string trackId = latterObjectInfoList[i].trackId;
        ObjectInfo latterObject = latterObjectInfoList[i];
        ObjectInfo previousObject {};
        for (size_t j = 0; j < previousObjectInfoList.size(); ++j) {
            if (trackId == previousObjectInfoList[j].trackId) {
                previousObject = previousObjectInfoList[j];
            }
        }
        ObjectInfo interpolatedObject {};
        interpolatedObject.trackId = trackId;
        interpolatedObject.x0 = previousObject.x0 + (latterObject.x0 - previousObject.x0) * offset;
        interpolatedObject.y0 = previousObject.y0 + (latterObject.y0 - previousObject.y0) * offset;
        interpolatedObject.x1 = previousObject.x1 + (latterObject.x1 - previousObject.x1) * offset;
        interpolatedObject.y1 = previousObject.y1 + (latterObject.y1 - previousObject.y1) * offset;
        interpolatedObjectInfoList.push_back(interpolatedObject);
    }
}

void MxpiFrameAlign::AlignFrameObjectInfo()
{
    if (objectListMap_.Size() >= ALIGN_NUM) {
        auto previousIter = objectListMap_.RBegin();
        previousIter--;
        auto lastIter = objectListMap_.RBegin();
        if (previousFrameId_ == 0) {
            previousFrameId_ = previousIter->first;
        }
        std::vector<ObjectInfo> previousObjectInfoList = previousIter->second;
        std::vector<ObjectInfo> objectInfoAllVector = lastIter->second;
        std::vector<ObjectInfo> objectInfoMatchVector;
        for (size_t i = 0; i < objectInfoAllVector.size(); ++i) {
            std::string trackId = objectInfoAllVector[i].trackId;
            if (HadTrackId(previousObjectInfoList, trackId)) {
                objectInfoMatchVector.push_back(objectInfoAllVector[i]);
            }
        }
        uint32_t frameId = lastIter->first;
        uint32_t previousFrameId = previousFrameId_;
        uint64_t frameStep = frameId - previousFrameId_;
        int stepNo = 0;
        while (previousFrameId_ < frameId) {
            if (previousFrameId_ == previousFrameId) {
                if (streamDataMap_.Find(previousFrameId_)) {
                    StreamData streamData = streamDataMap_.Pop(previousFrameId_);
                    streamData.sendFlag = true;
                    streamDataMap_.Insert(previousFrameId_, streamData);
                }
                previousFrameId_++;
                continue;
            }
            stepNo = previousFrameId_ - previousFrameId;
            float offset = float(stepNo) / float(frameStep);
            std::vector<ObjectInfo> interpolatedObjectInfoList;
            ObjectInfoInterpolated(interpolatedObjectInfoList, previousObjectInfoList, objectInfoMatchVector, offset);
            objectListMap_.Insert(previousFrameId_, interpolatedObjectInfoList);
            if (streamDataMap_.Find(previousFrameId_)) {
                StreamData streamData = streamDataMap_.Pop(previousFrameId_);
                streamData.sendFlag = true;
                streamDataMap_.Insert(previousFrameId_, streamData);
            }
            previousFrameId_++;
        }
    }
}

APP_ERROR MxpiFrameAlign::SendAlignFrame()
{
    for (auto iterFrame = streamDataMap_.Begin(); iterFrame != streamDataMap_.End();) {
        if (iterFrame->second.sendFlag) {
            uint32_t frameId = iterFrame->first;
            InputParam inputParam = {};
            inputParam.dataSize = 0;
            auto* mxpiBuffer = MxpiBufferManager::CreateHostBufferAndCopyData(inputParam);
            std::shared_ptr<MxpiWebDisplayDataList> webDisplayDataList = std::make_shared<MxpiWebDisplayDataList>();
            MxpiMetadataManager mxpiMetadataManager(*mxpiBuffer);
            APP_ERROR ret = GetWebDisplayData(webDisplayDataList, frameId);
            if (ret != APP_ERR_OK) {
                errorInfo_ << GetError(ret, elementName_) << "Failed to get web display data.";
                return ret;
            }
            ret = mxpiMetadataManager.AddProtoMetadata(elementName_,
                                                       std::static_pointer_cast<void>(webDisplayDataList));
            if (ret != APP_ERR_OK) {
                errorInfo_ << GetError(ret, elementName_) << "Failed to add metadata.";
                return ret;
            }
            SendData(0, *mxpiBuffer);
            break;
        } else {
            ++iterFrame;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR MxpiFrameAlign::GetWebDisplayData(std::shared_ptr<MxpiWebDisplayDataList> &webDisplayDataList,
                                            uint32_t &frameId)
{
    if (!objectListMap_.Find(frameId)) {
        LogError << "FrameId is invaild. " << "FrameId = " << frameId << " .";
        return APP_ERR_COMM_INNER;
    }
    auto objectList = objectListMap_.Pop(frameId);
    auto streamData = streamDataMap_.Pop(frameId);
    MxpiWebDisplayData* webDisplayData = webDisplayDataList->add_webdisplaydatavec();
    webDisplayData->CopyFrom(streamData.webDisplayData);
    for (size_t i = 0; i < objectList.size(); ++i) {
        MxpiBBox* boundingBox = webDisplayData->add_bbox_vec();
        boundingBox->set_x0(objectList[i].x0);
        boundingBox->set_y0(objectList[i].y0);
        boundingBox->set_x1(objectList[i].x1);
        boundingBox->set_y1(objectList[i].y1);
    }
    return APP_ERR_OK;
}

std::vector<std::string> MxpiFrameAlign::Split(const std::string &inString, char delimiter)
{
    std::vector<std::string> result;
    if (inString.empty()) {
        return result;
    }

    std::string::size_type fast = 0;
    std::string::size_type slow = 0;
    while ((fast = inString.find_first_of(delimiter, slow)) != std::string::npos) {
        result.push_back(inString.substr(slow, fast - slow));
        slow = inString.find_first_not_of(delimiter, fast);
    }

    if (slow != std::string::npos) {
        result.push_back(inString.substr(slow, fast - slow));
    }

    return result;
}

std::string &MxpiFrameAlign::Trim(std::string &str)
{
    str.erase(0, str.find_first_not_of(' '));
    str.erase(str.find_last_not_of(' ') + 1);

    return str;
}

std::vector<std::string> MxpiFrameAlign::SplitWithRemoveBlank(std::string &str, char rule)
{
    Trim(str);
    std::vector<std::string> strVec = Split(str, rule);
    for (size_t i = 0; i < strVec.size(); i++) {
        strVec[i] = Trim(strVec[i]);
    }
    return strVec;
}

void MxpiFrameAlign::SendThread()
{
    while (!sendStop_) {
        // send data if exist time is greater than maxSendTime_
        APP_ERROR ret = SendAlignFrame();
        if (ret != APP_ERR_OK) {
            LogError << errorInfo_.str();
            continue;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(intervalTime_));
    }
}

namespace {
MX_PLUGIN_GENERATE(MxpiFrameAlign)
}