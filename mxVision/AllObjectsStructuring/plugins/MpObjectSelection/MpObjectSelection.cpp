/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "MpObjectSelection.h"

#include <cmath>
#include <cfloat>
#include <boost/algorithm/string.hpp>
#include "MxTools/Proto/MxpiDataTypeDeleter.h"
#include "MxBase/CV/Core/DataType.h"

using namespace MxBase;
using namespace MxTools;

namespace {
    const int KEY_COUNT = 2;
    const int FRAME_COUNT_FOR_SEC = 10;
    const int MARGINRATE_COUNT = 3;
}

APP_ERROR MpObjectSelection::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "Start to initialize MpObjectSelection(" << elementName_ << ").";
    prePluginName_ = *std::static_pointer_cast<std::string>(configParamMap["dataSourceDetection"]);
    cropPluginName_ = *std::static_pointer_cast<std::string>(configParamMap["dataSourceImage"]);
    tmargin_ = *std::static_pointer_cast<float>(configParamMap["tmarginValue"]);
    weightMargin_ = *std::static_pointer_cast<float>(configParamMap["weightMargin"]);
    weightOcclude_ = *std::static_pointer_cast<float>(configParamMap["weightOcclude"]);
    weightSize_ = *std::static_pointer_cast<float>(configParamMap["weightSize"]);
    weightConf_ = *std::static_pointer_cast<float>(configParamMap["weightConf"]);
    std::string normRadius = *std::static_pointer_cast<std::string>(configParamMap["normRadius"]);
    trackTime_ = *std::static_pointer_cast<size_t>(configParamMap["trackTime"]);
    std::string keys = *std::static_pointer_cast<std::string>(configParamMap["outputKeys"]);
    if (keys != "") {
        boost::split(keysVec_, keys, boost::is_any_of(","), boost::token_compress_on);
    } else {
        LogError << "Using property key";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    if (normRadius.empty() || keysVec_.size() < KEY_COUNT) {
        errorInfo_ << GetError(APP_ERR_COMM_INVALID_PARAM, elementName_) << "Get norm radius failed.";
        LogError << errorInfo_.str();
        return APP_ERR_COMM_INVALID_PARAM;
    }
    GetNormRadius(normRadius);
    LogInfo << "End to initialize MpObjectSelection(" << elementName_ << ").";
    return APP_ERR_OK;
}

void MpObjectSelection::GetNormRadius(std::string& normRadius)
{
    size_t index = 0;
    while ((index = normRadius.find(' ', index)) != std::string::npos) {
        normRadius.erase(index, 1);
    }
    std::vector<std::string> normRadiusVec;
    boost::split(normRadiusVec, normRadius, boost::is_any_of(","), boost::token_compress_on);
    for (size_t i = 0; i < normRadiusVec.size(); i++) {
        std::istringstream iss(normRadiusVec[i]);
        float normRadiu;
        iss >> normRadiu;
        normRadius_.push_back(normRadiu);
    }
}

APP_ERROR MpObjectSelection::DeInit()
{
    LogInfo << "Start to deinitialize MpObjectSelection(" << elementName_ << ").";
    LogInfo << "End to deinitialize MpObjectSelection(" << elementName_ << ").";
    return APP_ERR_OK;
}

APP_ERROR MpObjectSelection::SetMxpiErrorInfo(const std::string pluginName, APP_ERROR errorCode,
    const std::string& errorText)
{
    InputParam inputParam = {};
    inputParam.mxpiMemoryType = MXPI_MEMORY_DVPP;
    inputParam.deviceId = deviceId_;
    inputParam.dataSize = 0;
    MxpiBuffer* mxpiBuffer = MxpiBufferManager::CreateDeviceBufferWithMemory(inputParam);
    if (mxpiBuffer == nullptr) {
        return APP_ERR_OK;
    }

    if (errorText == "") {
        return SendData(0, *mxpiBuffer);
    }
    return SendMxpiErrorInfo(*mxpiBuffer, pluginName, errorCode, errorText);
}

APP_ERROR MpObjectSelection::CheckInputBuffer(MxpiBuffer& motBuffer)
{
    errorInfo_.str("");

    MxpiMetadataManager mxpiMetadataManager(motBuffer);
    if (mxpiMetadataManager.GetErrorInfo() != nullptr) {
        LogWarn << "Input data is invalid, element(" << elementName_
                << ") plugin will not be executed rightly.";
        SetMxpiErrorInfo(elementName_, APP_ERR_COMM_INVALID_PARAM, "");
        MxpiBufferManager::DestroyBuffer(&motBuffer);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    auto metadataPtr = mxpiMetadataManager.GetMetadata(prePluginName_);
    if (metadataPtr == nullptr) {
        errorInfo_ << GetError(APP_ERR_COMM_INVALID_PARAM, elementName_) << "Get previous plugin pointer failed.";
        LogDebug << errorInfo_.str();
        SetMxpiErrorInfo(elementName_, APP_ERR_COMM_INVALID_PARAM, "");
        MxpiBufferManager::DestroyBuffer(&motBuffer);
        return APP_ERR_COMM_INVALID_PARAM;
    } else {
        auto metadataType = mxpiMetadataManager.GetMetadataWithType(prePluginName_, "MxpiTrackLetList");
        if (metadataType == nullptr) {
            errorInfo_ << GetError(APP_ERR_COMM_INVALID_PARAM, elementName_) << "Get the error type.";
            LogError << errorInfo_.str();
            SetMxpiErrorInfo(elementName_, APP_ERR_COMM_INVALID_PARAM, "Get the error type.");
            MxpiBufferManager::DestroyBuffer(&motBuffer);
            return APP_ERR_COMM_INVALID_PARAM;
        }
    }
    std::shared_ptr<MxpiTrackLetList> mxpiTrackLetList = std::static_pointer_cast<MxpiTrackLetList>(
        mxpiMetadataManager.GetMetadata(prePluginName_));
    if (mxpiTrackLetList == nullptr) {
        errorInfo_ << GetError(APP_ERR_COMM_INVALID_PARAM, elementName_) << "Get previous plugin pointer failed.";
        LogError << errorInfo_.str();
        SetMxpiErrorInfo(elementName_, APP_ERR_COMM_INVALID_PARAM, "Get previous plugin pointer failed.");
        MxpiBufferManager::DestroyBuffer(&motBuffer);
        return APP_ERR_COMM_INVALID_PARAM;
    } else if (mxpiTrackLetList->trackletvec_size() == 0) {
        errorInfo_ << GetError(APP_ERR_COMM_INVALID_PARAM, elementName_) << "Track list is null.";
        LogError << errorInfo_.str();
        MxpiBufferManager::DestroyBuffer(&motBuffer);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    return APP_ERR_OK;
}

APP_ERROR MpObjectSelection::Process(std::vector<MxpiBuffer*>& mxpiBuffer)
{
    LogDebug << "Begin to process MpObjectSelection(" << elementName_ << ").";
    MxpiBuffer *inputMxpiBuffer = mxpiBuffer[0];
    APP_ERROR ret = CheckInputBuffer(*inputMxpiBuffer);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    MxpiMetadataManager mxpiMetadataManager(*inputMxpiBuffer);
    std::shared_ptr<MxpiTrackLetList> mxpiTrackLetList = std::static_pointer_cast<MxpiTrackLetList>(
        mxpiMetadataManager.GetMetadata(prePluginName_));
    ret = TargetSelect(*inputMxpiBuffer, mxpiTrackLetList);
    if (ret != APP_ERR_OK) {
        SetMxpiErrorInfo(elementName_, APP_ERR_COMM_INVALID_PARAM, "Select target failed.");
        MxpiBufferManager::DestroyBuffer(inputMxpiBuffer);
        return ret;
    }
    MxpiBufferManager::DestroyBuffer(inputMxpiBuffer);
    LogDebug << "End to process MpObjectSelection(" << elementName_ << ").";
    return APP_ERR_OK;
}

APP_ERROR MpObjectSelection::TargetSelect(MxpiBuffer& buffer, std::shared_ptr<MxpiTrackLetList>& datalist)
{
    MxpiMetadataManager mxpiMetadataManager(buffer);
    MxpiFrame inputMxpiFrame = MxpiBufferManager::GetHostDataInfo(buffer);
    MxpiVisionList visionList = inputMxpiFrame.visionlist();
    channelId_ = inputMxpiFrame.frameinfo().channelid();
    int imageWidth = visionList.visionvec(0).visioninfo().width();
    int imageHeight = visionList.visionvec(0).visioninfo().height();
    while (!stackSet_.empty()) {
        stackSet_.pop();
    }
    APP_ERROR ret = PushDataToStack(buffer, *datalist, imageHeight);
    if (ret != APP_ERR_OK) {
        LogError << errorInfo_.str();
        return ret;
    }

    std::shared_ptr<MxpiVisionList> cropVisionList = std::static_pointer_cast<MxpiVisionList>(
        mxpiMetadataManager.GetMetadata(cropPluginName_));
    if (cropVisionList == nullptr) {
        ret = APP_ERR_COMM_INVALID_PARAM;
        errorInfo_ << GetError(ret, elementName_) << "Get previous plugin pointer failed.";
        LogError << errorInfo_.str();
        return ret;
    }
    ret = StartSelect(*cropVisionList, imageHeight, imageWidth);
    if (ret != APP_ERR_OK) {
        errorInfo_ << GetError(ret, elementName_) << "Select target failed.";
        LogError << errorInfo_.str();
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR MpObjectSelection::PushDataToStack(MxpiBuffer& buffer, MxpiTrackLetList& trackLetList, float yMax)
{
    LogDebug << "Start to push data to stack(" << elementName_ << ").";
    std::string cropParentName = "";
    for (int i = 0; i < trackLetList.trackletvec_size(); i++) {
        if (trackLetList.trackletvec(i).headervec_size() > 0) {
            cropParentName = trackLetList.trackletvec(i).headervec(0).datasource();
            break;
        }
    }
    MxpiMetadataManager mxpiMetadataManager(buffer);
    std::shared_ptr<MxpiObjectList> mxpiObjectList = std::static_pointer_cast<MxpiObjectList>(
        mxpiMetadataManager.GetMetadata(cropParentName));
    if (mxpiObjectList == nullptr) {
        errorInfo_ << GetError(APP_ERR_COMM_INVALID_PARAM, elementName_) << "Get object list failed.";
        LogError << errorInfo_.str();
        return APP_ERR_COMM_INVALID_PARAM;
    }
    APP_ERROR ret = PushObject(yMax, trackLetList, mxpiObjectList);
    if (ret != APP_ERR_OK) {
        LogError << errorInfo_.str();
        return ret;
    }
    LogDebug << "End to push data to stack(" << elementName_ << ").";
    return APP_ERR_OK;
}

APP_ERROR MpObjectSelection::PushObject(float yMax, MxTools::MxpiTrackLetList& trackLetList,
    std::shared_ptr<MxTools::MxpiObjectList>& mxpiObjectList)
{
    APP_ERROR ret = APP_ERR_OK;
    std::vector<TargetTrack> targetTrack = {};
    TargetTrack tmpTargetTrack = {};
    for (int i = 0; i < trackLetList.trackletvec_size(); i++) {
        if (trackLetList.trackletvec(i).trackflag() == LOST_OBJECT) {
            ret = CreatNewBuffer(trackLetList.trackletvec(i).trackid());
            if (ret != APP_ERR_OK) {
                LogError << errorInfo_.str();
                return ret;
            }
            continue;
        }
        int objectIndex = trackLetList.trackletvec(i).headervec(0).memberid();
        if (objectIndex >= mxpiObjectList->objectvec_size()) {
            errorInfo_ << GetError(APP_ERR_COMM_INVALID_POINTER, elementName_) << "Get object failed.";
            LogError << errorInfo_.str();
            return APP_ERR_COMM_INVALID_PARAM;
        }
        tmpTargetTrack.mxpiObject.CopyFrom(mxpiObjectList->objectvec(objectIndex));
        tmpTargetTrack.mxpiTrackLet.CopyFrom(trackLetList.trackletvec(i));
        tmpTargetTrack.channelId = channelId_;
        targetTrack.push_back(tmpTargetTrack);
    }
    std::map<int, int> tmpObject = {};
    for (size_t i = 0; i < targetTrack.size(); i++) {
        int tmpIndex = 0;
        float tmpYmax = yMax;
        for (size_t j = 0; j < targetTrack.size(); j++) {
            if (tmpObject.find(j) == tmpObject.end() && tmpYmax >= targetTrack[j].mxpiObject.y1()) {
                tmpYmax = targetTrack[j].mxpiObject.y1();
                tmpIndex = j;
            }
        }
        stackSet_.push(targetTrack[tmpIndex]);
        tmpObject[tmpIndex] = 1;
    }
    return APP_ERR_OK;
}

APP_ERROR MpObjectSelection::CreatNewBuffer(const int trackId, bool refresh)
{
    APP_ERROR ret;
    auto iter = targetTrack_.find(trackId);
    if (iter == targetTrack_.end()) {
        errorInfo_ << GetError(APP_ERR_COMM_INVALID_PARAM, elementName_) << "Get track id(" << trackId << ") failed.";
        LogError << errorInfo_.str();
        return APP_ERR_COMM_INVALID_PARAM;
    }
    InputParam inputParam = {keysVec_[0], deviceId_, (int)iter->second.data.size, iter->second.data.ptrData};
    inputParam.mxpiVisionInfo.CopyFrom(iter->second.mxpiVision.visioninfo());
    inputParam.mxpiFrameInfo.set_frameid(frameId_);
    inputParam.mxpiFrameInfo.set_channelid(iter->second.channelId);
    inputParam.mxpiMemoryType = MXPI_MEMORY_DVPP;
    MxpiBuffer* mxpiBuffer;
    if (!refresh) {
        mxpiBuffer = MxpiBufferManager::CreateDeviceBufferAndCopyData(inputParam);
    } else {
        mxpiBuffer = MxpiBufferManager::CreateDeviceBufferWithMemory(inputParam);
    }

    if (mxpiBuffer == nullptr) {
        LogError << "New buffer is null, trackId(" << trackId << "), dataptr("
                 << iter->second.data.ptrData << ").";
        ret = MxBase::MemoryHelper::MxbsFree(iter->second.data);
        targetTrack_.erase(iter);
        if (ret != APP_ERR_OK) {
            errorInfo_ << GetError(APP_ERR_COMM_INVALID_PARAM, elementName_) << "Failed to free data.";
            LogError << errorInfo_.str();
            return ret;
        }
        return APP_ERR_OK;
    }

    ret = AddObjectList(*mxpiBuffer, iter);
    if (ret != APP_ERR_OK) {
        LogError << errorInfo_.str();
        return ret;
    }
    if (!refresh) {
        return APP_ERR_OK;
    }
    targetTrack_.erase(iter);
    return APP_ERR_OK;
}

APP_ERROR MpObjectSelection::AddObjectList(MxTools::MxpiBuffer& buffer, std::map<int, TargetTrack>::iterator& iter)
{
    std::shared_ptr<MxTools::MxpiObjectList> objectList = std::make_shared<MxTools::MxpiObjectList>();
    MxTools::MxpiObject* objectData = objectList->add_objectvec();
    objectData->set_x0(iter->second.mxpiObject.x0());
    objectData->set_y0(iter->second.mxpiObject.y0());
    objectData->set_x1(iter->second.mxpiObject.x1());
    objectData->set_y1(iter->second.mxpiObject.y1());
    MxTools::MxpiClass* classInfo = objectData->add_classvec();
    classInfo->set_classid(iter->second.mxpiObject.classvec(0).classid());
    classInfo->set_classname(iter->second.mxpiObject.classvec(0).classname());
    classInfo->set_confidence(iter->second.mxpiObject.classvec(0).confidence());
    MxTools::MxpiMetaHeader* header = objectData->add_headervec();
    header->set_datasource(pluginName_);
    header->set_memberid(0);
    MxTools::MxpiMetadataManager mxpiMetadataManager(buffer);
    APP_ERROR ret = mxpiMetadataManager.AddProtoMetadata(keysVec_[1], objectList);
    if (ret != APP_ERR_OK) {
        errorInfo_ << GetError(APP_ERR_COMM_INVALID_PARAM, elementName_)
                   << "Add proto metadata(" << keysVec_[1] << ") failed.";
        LogError << errorInfo_.str();
        return ret;
    }
    SendData(0, buffer); // Send the data to downstream plugin
    frameId_++;
    return APP_ERR_OK;
}

APP_ERROR MpObjectSelection::StartSelect(MxpiVisionList& cropVisionList, int imageHeight, int imageWidth)
{
    LogDebug << "Start to select(" << elementName_ << ").";
    APP_ERROR ret = APP_ERR_OK;
    frontSet_.clear();
    while (stackSet_.size() > 0) {
        TargetTrack targetTrack = stackSet_.top();
        targetTrack.imageHeight = imageHeight;
        targetTrack.imageWidth = imageWidth;
        ret = GetOccludeScore(targetTrack);
        if (ret != APP_ERR_OK) {
            LogError << errorInfo_.str();
            return ret;
        }
        for (int i = 0; i < cropVisionList.visionvec_size(); i++) {
            if (cropVisionList.visionvec(i).headervec(0).memberid() ==
                targetTrack.mxpiTrackLet.headervec(0).memberid()) {
                targetTrack.mxpiVision.CopyFrom(cropVisionList.visionvec(i));
                break;
            }
        }
        targetTrack.data.type = MxBase::MemoryData::MEMORY_DVPP;
        targetTrack.data.size = targetTrack.mxpiVision.visiondata().datasize();
        targetTrack.data.deviceId = deviceId_;
        MemoryData src;
        src.type = MxBase::MemoryData::MEMORY_DVPP;
        src.size = targetTrack.mxpiVision.visiondata().datasize();
        src.ptrData = (void*)targetTrack.mxpiVision.visiondata().dataptr();
        src.deviceId = deviceId_;
        ret = MxBase::MemoryHelper::MxbsMallocAndCopy(targetTrack.data, src);
        if (ret != APP_ERR_OK) {
            errorInfo_ << GetError(APP_ERR_COMM_INVALID_PARAM, elementName_) << "Failed to malloc and copy data.";
            LogError << errorInfo_.str();
            return ret;
        }
        targetTrack.age = 0;
        ret = RefleshData(targetTrack);
        if (ret != APP_ERR_OK) {
            LogError << errorInfo_.str();
            return ret;
        }
        stackSet_.pop();
    }
    LogDebug << "End to select(" << elementName_ << ").";
    return APP_ERR_OK;
}

APP_ERROR MpObjectSelection::RefleshData(TargetTrack& targetTrack)
{
    APP_ERROR ret;
    auto iter = targetTrack_.find(targetTrack.mxpiTrackLet.trackid());
    if (iter != targetTrack_.end()) {
        if (iter->second.score < targetTrack.score) {
            iter->second.score = targetTrack.score;
            iter->second.marginScore = targetTrack.marginScore;
            iter->second.occludeScore = targetTrack.occludeScore;
            iter->second.imageHeight = targetTrack.imageHeight;
            iter->second.imageWidth = targetTrack.imageWidth;
            iter->second.channelId = targetTrack.channelId;
            iter->second.mxpiVision.CopyFrom(targetTrack.mxpiVision);
            ret = MxBase::MemoryHelper::MxbsFree(iter->second.data);
            if (ret != APP_ERR_OK) {
                errorInfo_ << GetError(APP_ERR_COMM_INVALID_PARAM, elementName_) << "Failed to free data.";
                LogError << errorInfo_.str();
                return ret;
            }
            iter->second.data = targetTrack.data;
            iter->second.mxpiObject.CopyFrom(targetTrack.mxpiObject);
        } else {
            ret = MxBase::MemoryHelper::MxbsFree(targetTrack.data);
            if (ret != APP_ERR_OK) {
                errorInfo_ << GetError(APP_ERR_COMM_INVALID_PARAM, elementName_) << "Failed to free data.";
                LogError << errorInfo_.str();
                return ret;
            }
        }
        iter->second.mxpiTrackLet.CopyFrom(targetTrack.mxpiTrackLet);

        if (CheckSendData(targetTrack.mxpiTrackLet.trackid())) {
            ret = CreatNewBuffer(targetTrack.mxpiTrackLet.trackid(), false);
            if (ret != APP_ERR_OK) {
                LogError << errorInfo_.str();
                return ret;
            }
        }
    } else {
        targetTrack.age = targetTrack.mxpiTrackLet.hits();
        targetTrack_[targetTrack.mxpiTrackLet.trackid()] = targetTrack;
    }
    return APP_ERR_OK;
}

APP_ERROR MpObjectSelection::GetPositionScore(const MxpiObject& mxpiObject, TargetTrack& targetTrack)
{
    if (targetTrack.imageHeight <= 0 || targetTrack.imageWidth <= 0) {
        errorInfo_ << GetError(APP_ERR_COMM_INVALID_PARAM, elementName_) << "Input image width is ("
                   << targetTrack.imageWidth << "), height is (" << targetTrack.imageHeight << "). Please check it.";
        LogError << errorInfo_.str();
        return APP_ERR_COMM_INVALID_PARAM;
    }
    int marginRateY = (targetTrack.imageHeight - mxpiObject.y1()) / targetTrack.imageHeight;
    marginRateY = (marginRateY > tmargin_) ? tmargin_ : marginRateY;
    int marginRateXRight = (targetTrack.imageWidth - mxpiObject.x1()) / targetTrack.imageWidth;
    int marginRateXLeft = mxpiObject.x0() / targetTrack.imageWidth;
    marginRateXRight = (marginRateXRight > tmargin_) ? tmargin_ : marginRateXRight;
    marginRateXLeft = (marginRateXLeft > tmargin_) ? tmargin_ : marginRateXLeft;
    marginRateY = marginRateY / tmargin_;
    marginRateXRight = marginRateXRight / tmargin_;
    marginRateXLeft = marginRateXLeft / tmargin_;
    targetTrack.marginScore = (marginRateY + marginRateXRight + marginRateXLeft) / MARGINRATE_COUNT;
    return APP_ERR_OK;
}

APP_ERROR MpObjectSelection::GetOccludeScore(TargetTrack& targetTrack)
{
    APP_ERROR ret;
    float occludeScore = 0.0;
    float targetArea = (targetTrack.mxpiObject.x1() - targetTrack.mxpiObject.x0()) *
                       (targetTrack.mxpiObject.y1() - targetTrack.mxpiObject.y0());
    if (fabs(targetArea) < FLT_EPSILON) {
        errorInfo_ << GetError(APP_ERR_COMM_INVALID_PARAM, elementName_) << ", Input image width is ("
                   << targetTrack.imageWidth << "), height is (" << targetTrack.imageHeight << "). Please check it.";
        LogError << errorInfo_.str();
        return APP_ERR_COMM_INVALID_PARAM;
    }
    for (size_t i = 0; i < frontSet_.size(); i++) {
        if (targetTrack.mxpiObject.x1() <= frontSet_[i].mxpiObject.x0() ||
            targetTrack.mxpiObject.x0() >= frontSet_[i].mxpiObject.x1() ||
            targetTrack.mxpiObject.y1() >= frontSet_[i].mxpiObject.y0()) {
            continue;
        } else {
            float tmpXmin = (targetTrack.mxpiObject.x0() > frontSet_[i].mxpiObject.x0()) ?
                            targetTrack.mxpiObject.x0() : frontSet_[i].mxpiObject.x0();
            float tmpXmax = (targetTrack.mxpiObject.x1() < frontSet_[i].mxpiObject.x1()) ?
                            targetTrack.mxpiObject.x0() : frontSet_[i].mxpiObject.x0();
            float intersection = (tmpXmax - tmpXmin) *
                                 (targetTrack.mxpiObject.y1() - frontSet_[i].mxpiObject.y0());
            occludeScore = (occludeScore > (intersection / targetArea)) ? occludeScore : (intersection / targetArea);
        }
    }
    targetTrack.occludeScore = occludeScore;
    frontSet_.push_back(targetTrack);
    ret = GetPositionScore(targetTrack.mxpiObject, targetTrack);
    if (ret != APP_ERR_OK) {
        LogError << errorInfo_.str();
        return ret;
    }
    float imageWidth = targetTrack.mxpiObject.x1() - targetTrack.mxpiObject.x0();
    float imageHeight = targetTrack.mxpiObject.y1() - targetTrack.mxpiObject.y0();
    float imageArea = imageWidth * imageHeight;
    if ((size_t)targetTrack.mxpiObject.classvec(0).classid() < normRadius_.size() &&
        fabs(normRadius_[targetTrack.mxpiObject.classvec(0).classid()]) > FLT_EPSILON) {
        targetTrack.sizeScore = sqrt(imageArea) / normRadius_[targetTrack.mxpiObject.classvec(0).classid()];
    }

    targetTrack.score = weightMargin_ * targetTrack.marginScore + weightOcclude_ * targetTrack.occludeScore +
                        weightSize_ * targetTrack.sizeScore + weightConf_ * targetTrack.confScore;
    return APP_ERR_OK;
}

bool MpObjectSelection::CheckSendData(const int trackId)
{
    auto iter = targetTrack_.find(trackId);
    if (iter == targetTrack_.end()) {
        errorInfo_ << GetError(APP_ERR_COMM_INVALID_PARAM, elementName_)
                   << " Get track id (" << trackId << ") failed.";
        LogError << errorInfo_.str();
        return false;
    }
    if (trackTime_ > 0 && iter->second.age++ > trackTime_ * FRAME_COUNT_FOR_SEC) {
        iter->second.age = 0;
        return true;
    }
    return false;
}

MxpiPortInfo MpObjectSelection::DefineInputPorts()
{
    MxpiPortInfo inputPortInfo;
    std::vector<std::vector<std::string>> value = {{"ANY"}};
    GenerateStaticInputPortsInfo(value, inputPortInfo);

    return inputPortInfo;
}

MxpiPortInfo MpObjectSelection::DefineOutputPorts()
{
    MxpiPortInfo outputPortInfo;
    std::vector<std::vector<std::string>> value = {{"ANY"}};
    GenerateStaticOutputPortsInfo(value, outputPortInfo);

    return outputPortInfo;
}

std::vector<std::shared_ptr<void>> MpObjectSelection::DefineProperties()
{
    std::vector<std::shared_ptr<void>> properties;

    auto dataSourceDetection = std::make_shared<ElementProperty<std::string>>(ElementProperty<std::string> {
        STRING, "dataSourceDetection", "name", "The name of detection data source", "", "", ""
    });
    auto dataSourceImage = std::make_shared<ElementProperty<std::string>>(ElementProperty<std::string> {
        STRING, "dataSourceImage", "name", "The name of image data source", "", "", ""
    });
    auto tmarginValue = std::make_shared<ElementProperty<float>>(ElementProperty<float> {
        FLOAT, "tmarginValue", "value", "the value of tmargin", 0.1, 0.0, 0.2
    });
    auto weightMargin = std::make_shared<ElementProperty<float>>(ElementProperty<float> {
        FLOAT, "weightMargin", "value", "the value of weight margin", 0.1, 0.0, 0.5
    });
    auto weightOcclude = std::make_shared<ElementProperty<float>>(ElementProperty<float> {
        FLOAT, "weightOcclude", "value", "the value of weight occlude", 0.1, 0.0, 0.5
    });
    auto weightSize = std::make_shared<ElementProperty<float>>(ElementProperty<float> {
        FLOAT, "weightSize", "value", "the value of weight size", 0.1, 0.0, 0.5
    });
    auto weightConf = std::make_shared<ElementProperty<float>>(ElementProperty<float> {
        FLOAT, "weightConf", "value", "the value of weight conf", 0.1, 0.0, 0.5
    });
    auto trackTime = std::make_shared<ElementProperty<int>>(ElementProperty<int> {
        INT, "trackTime", "value", "the value of track time", 5, 0, 10000
    });
    auto normRadius = std::make_shared<ElementProperty<std::string>>(ElementProperty<std::string> {
        STRING, "normRadius", "norm radius", "the value of norm radius", "", "", ""
    });
    auto keys = std::make_shared<ElementProperty<std::string>>(ElementProperty<std::string> {
        STRING, "outputKeys", "outputKeys", "keys for add meta data", "", "", ""
    });
    properties.push_back(dataSourceDetection);
    properties.push_back(dataSourceImage);
    properties.push_back(tmarginValue);
    properties.push_back(weightMargin);
    properties.push_back(weightOcclude);
    properties.push_back(weightSize);
    properties.push_back(weightConf);
    properties.push_back(normRadius);
    properties.push_back(trackTime);
    properties.push_back(keys);
    return properties;
}

namespace {
MX_PLUGIN_GENERATE(MpObjectSelection)
}
