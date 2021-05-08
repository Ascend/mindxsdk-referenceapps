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

#include "MxpiFaceSelection.h"

#include <cmath>
#include "MxBase/Log/Log.h"

using namespace MxBase;
using namespace MxTools;

namespace {
const int KEY_POINTS_VEC_SIZE = 15;
const int FACE_KEY_POINT = 5;
}

APP_ERROR MxpiFaceSelection::Init(std::map<std::string, std::shared_ptr<void>> &configParamMap)
{
    LogInfo << "Begin to initialize MxpiFaceSelection(" << pluginName_ << ").";
    // Set previous MOT plugin name
    std::shared_ptr<std::string> trackedParentName = std::static_pointer_cast<std::string>(
        configParamMap["dataSourceDetection"]);
    trackedParentName_ = *trackedParentName;
    // Set previous face key point plugin name
    std::shared_ptr<std::string> keyPointParentName = std::static_pointer_cast<std::string>(
        configParamMap["dataSourceKeyPoint"]);
    keyPointParentName_ = *keyPointParentName;
    // Set weight of face key point score
    std::shared_ptr<float> keyPointWeight = std::static_pointer_cast<float>(configParamMap["keyPointWeight"]);
    keyPointWeight_ = *keyPointWeight;
    // Set weight of face pose score
    std::shared_ptr<float> eulerWeight = std::static_pointer_cast<float>(configParamMap["eulerWeight"]);
    eulerWeight_ = *eulerWeight;
    // Set weight of face size score
    std::shared_ptr<float> faceSizeWeight = std::static_pointer_cast<float>(configParamMap["faceSizeWeight"]);
    faceSizeWeight_ = *faceSizeWeight;
    // Set min face score threshold
    std::shared_ptr<float> minScoreThreshold = std::static_pointer_cast<float>(configParamMap["minScoreThreshold"]);
    minScoreThreshold_ = *minScoreThreshold;
    // Set max send age of face selection
    std::shared_ptr<uint> maxAge = std::static_pointer_cast<uint>(configParamMap["maxAge"]);
    maxAge_ = *maxAge;
    LogInfo << "End to initialize MxpiFaceSelection(" << pluginName_ << ").";
    return APP_ERR_OK;
}

APP_ERROR MxpiFaceSelection::DeInit()
{
    LogInfo << "Begin to deinitialize MxpiFaceSelection.";
    LogInfo << "End to deinitialize MxpiFaceSelection.";
    return APP_ERR_OK;
}

APP_ERROR MxpiFaceSelection::Process(std::vector<MxTools::MxpiBuffer*> &mxpiBuffer)
{
    LogDebug << "Begin to process MxpiFaceSelection(" << pluginName_ << ").";
    // Get mxpiBuffer from first import port
    MxpiBuffer* inputBuffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManagerFirst(*inputBuffer);
    errorInfo_.str("");
    MxpiErrorInfo mxpiErrorInfo;
    auto errorInfoPtr = mxpiMetadataManagerFirst.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        LogDebug << "WARNING. Input data is invalid, element(" << pluginName_
                 << ") plugin will not be executed rightly.";
        SendData(0, *inputBuffer);
        return APP_ERR_OK;
    }
    APP_ERROR ret = ErrorProcess(*inputBuffer);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    ret = CheckMetadataType(*inputBuffer);
    if (ret != APP_ERR_OK) {
        LogError << errorInfo_.str();
        SendMxpiErrorInfo(*inputBuffer, pluginName_, ret, errorInfo_.str());
        return ret;
    }
    // Get tracking and face key point from mxpiBuffer of first and second import port
    std::vector<FaceObject> faceObjectQueue;
    ret = GetPrePluginsResult(*inputBuffer, faceObjectQueue);
    if (ret != APP_ERR_OK) {
        LogError << errorInfo_.str();
        SendMxpiErrorInfo(*inputBuffer, pluginName_, ret, errorInfo_.str());
        return ret;
    }
    isUsedBuffer_ = false;
    FaceQualityEvaluation(faceObjectQueue, *inputBuffer);
    ret = GetFaceSelectionResult();
    if (ret != APP_ERR_OK) {
        LogError << errorInfo_.str();
        SendMxpiErrorInfo(*inputBuffer, pluginName_, ret, errorInfo_.str());
        return APP_ERR_OK;
    }
    if (!isUsedBuffer_) {
        MxpiBufferManager::DestroyBuffer(inputBuffer);
    }
    for (auto iter = bufferMap_.begin(); iter != bufferMap_.end(); ++iter) {
        if (iter->second.ref == 0) {
            iter->second.mxpiBuffer = nullptr;
            iter = bufferMap_.erase(iter);
        }
    }
    LogDebug << "End to process MxpiFaceSelection(" << pluginName_ << ").";
    return APP_ERR_OK;
}

APP_ERROR MxpiFaceSelection::CheckMetadataType(MxTools::MxpiBuffer &inputBuffer)
{
    MxpiMetadataManager mxpiMetadataManager(inputBuffer);
    auto trackedRes = mxpiMetadataManager.GetMetadataWithType(trackedParentName_, "MxpiTrackLetList");
    if (trackedRes == nullptr) {
        errorInfo_ << GetError(APP_ERR_PROTOBUF_NAME_MISMATCH, elementName_)
                   << "Not a MxpiTrackLetList object.";
        return APP_ERR_PROTOBUF_NAME_MISMATCH;
    }

    auto keyPointRes = mxpiMetadataManager.GetMetadataWithType(keyPointParentName_, "MxpiKeyPointAndAngleList");
    if (keyPointRes == nullptr) {
        errorInfo_ << GetError(APP_ERR_PROTOBUF_NAME_MISMATCH, elementName_)
                   << "Not a MxpiKeyPointAndAngleList object.";
        return APP_ERR_PROTOBUF_NAME_MISMATCH;
    }
    return APP_ERR_OK;
}

APP_ERROR MxpiFaceSelection::ErrorProcess(MxTools::MxpiBuffer &inputBuffer)
{
    MxpiMetadataManager mxpiMetadataManager(inputBuffer);
    if (mxpiMetadataManager.GetMetadata(trackedParentName_) == nullptr) {
        LogDebug << "Failed to get object tracked result.";
        gst_buffer_ref((GstBuffer*) inputBuffer.buffer);
        auto* tmpBuffer = new(std::nothrow) MxpiBuffer {inputBuffer.buffer};
        SendData(1, *tmpBuffer);
        SendData(0, inputBuffer);
        return APP_ERR_COMM_INVALID_POINTER;
    }

    if (mxpiMetadataManager.GetMetadata(keyPointParentName_) == nullptr) {
        LogDebug << "Failed to get face key points result.";
        gst_buffer_ref((GstBuffer*) inputBuffer.buffer);
        auto* tmpBuffer = new(std::nothrow) MxpiBuffer {inputBuffer.buffer};
        SendData(1, *tmpBuffer);
        SendData(0, inputBuffer);
        return APP_ERR_COMM_INVALID_POINTER;
    }

    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiFaceSelection::DefineProperties()
{
    std::vector<std::shared_ptr<void>> properties;
    auto keyPointWeight = std::make_shared<ElementProperty<float>>(ElementProperty<float> {
        FLOAT, "keyPointWeight", "keyPointWeight", "weight of key point score", 1.f, 0.f, 1.f
    });
    auto eulerWeight = std::make_shared<ElementProperty<float>>(ElementProperty<float> {
        FLOAT, "eulerWeight", "eulerWeight", "weight of face euler angles score", 1.f, 0.f, 1.f
    });
    auto faceSizeWeight = std::make_shared<ElementProperty<float>>(ElementProperty<float> {
        FLOAT, "faceSizeWeight", "faceSizeWeight", "weight of big face score", 1.f, 0.f, 1.f
    });
    auto minScoreThreshold = std::make_shared<ElementProperty<float>>(ElementProperty<float> {
        FLOAT, "minScoreThreshold", "bigFaceWeight", "min face total score threshold", 4.f, 0.f, 10.f
    });
    auto trackedParentName = std::make_shared<ElementProperty<std::string>>(ElementProperty<std::string> {
        STRING, "dataSourceDetection", "dataSourceDetection", "the key of detection data source", ""
    });
    auto keyPointParentName = std::make_shared<ElementProperty<std::string>>(ElementProperty<std::string> {
        STRING, "dataSourceKeyPoint", "dataSourceKeyPoint", "the key of face key point data source", ""
    });
    auto maxAge = std::make_shared<ElementProperty<uint>>(ElementProperty<uint> {
        UINT, "maxAge", "maxAge", "Max age for stopping face selection", 500, 1, 1000
    });
    properties.push_back(keyPointWeight);
    properties.push_back(eulerWeight);
    properties.push_back(faceSizeWeight);
    properties.push_back(minScoreThreshold);
    properties.push_back(trackedParentName);
    properties.push_back(keyPointParentName);
    properties.push_back(maxAge);
    return properties;
}

MxpiPortInfo MxpiFaceSelection::DefineInputPorts()
{
    MxpiPortInfo inputPortInfo;
    std::vector<std::vector<std::string>> value = {{"ANY"}};
    GenerateStaticInputPortsInfo(value, inputPortInfo);

    return inputPortInfo;
}

MxpiPortInfo MxpiFaceSelection::DefineOutputPorts()
{
    MxpiPortInfo outputPortInfo;
    std::vector<std::vector<std::string>> value = {{"ANY"}, {"ANY"}};
    GenerateStaticOutputPortsInfo(value, outputPortInfo);

    return outputPortInfo;
}

APP_ERROR MxpiFaceSelection::GetPrePluginsResult(MxpiBuffer &inputBuffer, std::vector<FaceObject> &faceObjectQueue)
{
    APP_ERROR ret = APP_ERR_COMM_INVALID_POINTER;
    MxpiMetadataManager mxpiMetadataManager(inputBuffer);
    MxpiFrame frameData = MxTools::MxpiBufferManager::GetDeviceDataInfo(inputBuffer);
    uint32_t frameId = frameData.frameinfo().frameid();
    uint32_t channelId = frameData.frameinfo().channelid();
    // Get object tracking resulut from the buffer
    std::shared_ptr<MxpiTrackLetList> trackLetList = std::static_pointer_cast<MxpiTrackLetList>(
        mxpiMetadataManager.GetMetadata(trackedParentName_));
    if (trackLetList->trackletvec_size() == 0) {
        errorInfo_ << GetError(ret, pluginName_) << "Failed to get object tracked result.";
        return ret;
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
        errorInfo_ << GetError(ret, pluginName_) << "Failed to get detection result.";
        return ret;
    }
    // Get face key point result list from the buffer
    std::shared_ptr<MxpiKeyPointAndAngleList> keyPointAndAngleList = std::static_pointer_cast<MxpiKeyPointAndAngleList>(
        mxpiMetadataManager.GetMetadata(keyPointParentName_));
    if (keyPointAndAngleList->keypointandanglevec_size() == 0) {
        errorInfo_ << GetError(ret, pluginName_) << "Failed to get face key points result.";
        return ret;
    }
    for (uint32_t i = 0; i < trackLetList->trackletvec_size(); ++i) {
        FaceObject faceObject {trackedParentName_, i, frameId, channelId, 0, trackLetList->trackletvec(i)};
        if (trackLetList->trackletvec(i).headervec_size() == 0) {
            faceObjectQueue.push_back(faceObject);
            continue;
        }
        int32_t memberId = trackLetList->trackletvec(i).headervec(0).memberid();
        if (memberId >= objectList->objectvec_size() || memberId >= keyPointAndAngleList->keypointandanglevec_size()) {
            faceObjectQueue.push_back(faceObject);
            continue;
        }
        // Get object detection and face key point result by memberId
        faceObject.detectInfo = objectList->objectvec(memberId);
        faceObject.keyPointAndAngle = keyPointAndAngleList->keypointandanglevec(memberId);
        faceObjectQueue.push_back(faceObject);
    }
    return APP_ERR_OK;
}

float MxpiFaceSelection::CalKeyPointScore(const FaceObject &faceObject)
{
    float score = 0.f;
    if (faceObject.keyPointAndAngle.keypointsvec_size() == 0) {
        LogWarn << "WARNING. Key points vector size is zero, trackId is: " << faceObject.trackLet.trackid() << ".";
        return score;
    }
    if (faceObject.keyPointAndAngle.keypointsvec_size() != KEY_POINTS_VEC_SIZE) {
        LogWarn << "WARNING. Key points vector size is invalid, trackId is: " << faceObject.trackLet.trackid() << ".";
        return score;
    }
    float elementScoreLimit = 0.2;
    int IndexOffset = 10;
    for (int i = 0; i < FACE_KEY_POINT; ++i) {
        float tmpScore = faceObject.keyPointAndAngle.keypointsvec(i + IndexOffset);
        score += ((tmpScore > elementScoreLimit) ? elementScoreLimit : tmpScore);
    }
    return score;
}

float MxpiFaceSelection::CalEulerScore(const FaceObject &faceObject)
{
    if (faceObject.keyPointAndAngle.keypointsvec_size() == 0) {
        LogWarn << "WARNING. Key points vector size is zero, trackId is: " << faceObject.trackLet.trackid() << ".";
        return 0.f;
    }
    uint32_t degree90 = 90;
    uint32_t pitchConstant = 6;
    float yaw = faceObject.keyPointAndAngle.angleyaw();
    float pitch = faceObject.keyPointAndAngle.anglepitch();
    float roll = faceObject.keyPointAndAngle.angleroll();
    pitch = (pitch > pitchConstant) ? pitch - pitchConstant : 0;
    return (degree90 - yaw) / degree90 + (degree90 - pitch) / degree90 + (degree90 - roll) / degree90;
}

float MxpiFaceSelection::CalFaceSizeScore(const FaceObject &faceObject)
{
    float width = faceObject.detectInfo.x1() - faceObject.detectInfo.x0();
    float height = faceObject.detectInfo.y1() - faceObject.detectInfo.y0();
    uint32_t maxFaceHW = 60;
    uint32_t normFaceHW = 50;
    float faceStretchRatio = 1.2;
    float faceScoreConstant = 3600.0;
    width = (width > normFaceHW) ? maxFaceHW : (width * faceStretchRatio);
    height = (height > normFaceHW) ? maxFaceHW : (height * faceStretchRatio);
    return 1 - std::fabs(maxFaceHW - width) * std::fabs(maxFaceHW - height) / faceScoreConstant;
}

float MxpiFaceSelection::CalTotalScore(const FaceObject &faceObject)
{
    float keyPointScore = CalKeyPointScore(faceObject);
    float eulerScore = CalEulerScore(faceObject);
    float faceSizeScore = CalFaceSizeScore(faceObject);
    float score = keyPointWeight_ * keyPointScore + eulerWeight_ * eulerScore + faceSizeWeight_ * faceSizeScore;
    return score;
}

void MxpiFaceSelection::FaceQualityEvaluation(std::vector<FaceObject> &faceObjectQueue, MxpiBuffer &buffer)
{
    uint32_t frameId = faceObjectQueue[0].frameId;
    BufferManager bufferManager {0, nullptr};
    std::vector<uint32_t> trackIdVec;
    for (size_t i = 0; i < faceObjectQueue.size(); ++i) {
        uint32_t trackId = faceObjectQueue[i].trackLet.trackid();
        if (faceObjectQueue[i].trackLet.trackflag() == LOST_OBJECT &&
            qualityAssessmentMap_.find(trackId) != qualityAssessmentMap_.end()) {
            qualityAssessmentMap_[trackId].trackLet.set_trackflag(LOST_OBJECT);
            continue;
        }
        if (bufferMap_.find(frameId) != bufferMap_.end()) {
            LogDebug << "FrameId is existed.";
            continue;
        }
        float score = CalTotalScore(faceObjectQueue[i]);
        if (score <= minScoreThreshold_) {
            continue;
        }
        faceObjectQueue[i].score = score;
        if (qualityAssessmentMap_.find(trackId) == qualityAssessmentMap_.end()) {
            qualityAssessmentMap_[trackId] = faceObjectQueue[i];
            bufferManager.ref++;
            trackIdVec.push_back(trackId);
        } else {
            if (qualityAssessmentMap_[trackId].score < score) {
                uint32_t oldFrameId = qualityAssessmentMap_[trackId].frameId;
                auto iter = std::find(bufferMap_[oldFrameId].trackIdVec.begin(),
                                      bufferMap_[oldFrameId].trackIdVec.end(), trackId);
                if (iter != bufferMap_[oldFrameId].trackIdVec.end()) {
                    bufferMap_[oldFrameId].trackIdVec.erase(iter);
                    bufferMap_[oldFrameId].ref--;
                }
                if (bufferMap_[oldFrameId].ref == 0) {
                    MxpiBufferManager::DestroyBuffer(bufferMap_[oldFrameId].mxpiBuffer);
                }
                qualityAssessmentMap_[trackId] = faceObjectQueue[i];
                bufferManager.ref++;
                trackIdVec.push_back(trackId);
            }
        }
    }
    if (bufferManager.ref > 0) {
        isUsedBuffer_ = true;
        bufferManager.mxpiBuffer = &buffer;
        bufferManager.trackIdVec = trackIdVec;
        bufferMap_[frameId] = bufferManager;
    }
}

APP_ERROR MxpiFaceSelection::GetFaceSelectionResult()
{
    auto iter = qualityAssessmentMap_.begin();
    while (iter != qualityAssessmentMap_.end()) {
        if (iter->second.trackLet.trackflag() == LOST_OBJECT) {
            // Get MxpiObjectList result
            auto objectList = std::make_shared<MxpiObjectList>();
            GetObjectListResult(iter, objectList);
            // Get MxpiKeyPointAndAngleList result
            auto keyPointAndAngleList = std::make_shared<MxpiKeyPointAndAngleList>();
            GetKeyPointResult(iter, keyPointAndAngleList);
            APP_ERROR ret = SendSelectionDate(iter, objectList, keyPointAndAngleList);
            if (ret != APP_ERR_OK) {
                return ret;
            }
            iter = qualityAssessmentMap_.erase(iter);
        } else {
            ++iter;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR MxpiFaceSelection::SendSelectionDate(std::map<uint32_t, FaceObject>::iterator &iter,
                                               std::shared_ptr<MxTools::MxpiObjectList> &objectList,
                                               std::shared_ptr<MxTools::MxpiKeyPointAndAngleList> &keyPointAndAngleList)
{
    uint32_t frameId = iter->second.frameId;
    uint32_t trackId = iter->second.trackLet.trackid();
    uint32_t channelId = iter->second.channelId;
    if (bufferMap_[frameId].ref > 1) {
        MxpiFrame frameData = MxTools::MxpiBufferManager::GetDeviceDataInfo(*bufferMap_[frameId].mxpiBuffer);
        InputParam inputParam = {
            pluginName_, deviceId_, frameData.visionlist().visionvec(0).visiondata().datasize(),
            (void*) frameData.visionlist().visionvec(0).visiondata().dataptr()
        };
        inputParam.mxpiVisionInfo = frameData.visionlist().visionvec(0).visioninfo();
        inputParam.mxpiFrameInfo.set_frameid(frameId);
        inputParam.mxpiFrameInfo.set_channelid(channelId);
        inputParam.mxpiMemoryType = MXPI_MEMORY_DVPP;
        MxpiBuffer* mxpiBuffer = MxpiBufferManager::CreateDeviceBufferAndCopyData(inputParam);
        APP_ERROR ret = AddMetaData(*mxpiBuffer, objectList, keyPointAndAngleList);
        if (ret != APP_ERR_OK) {
            return ret;
        }
        gst_buffer_ref((GstBuffer*) mxpiBuffer->buffer);
        auto* tmpBuffer = new(std::nothrow) MxpiBuffer {mxpiBuffer->buffer};
        if (tmpBuffer == nullptr) {
            return APP_ERR_COMM_ALLOC_MEM;
        }
        SendData(0, *tmpBuffer);
        SendData(1, *mxpiBuffer);
    } else if (bufferMap_[frameId].ref == 1) {
        APP_ERROR ret = AddMetaData(*bufferMap_[frameId].mxpiBuffer, objectList, keyPointAndAngleList);
        if (ret != APP_ERR_OK) {
            return ret;
        }
        gst_buffer_ref((GstBuffer*) bufferMap_[frameId].mxpiBuffer->buffer);
        auto* tmpBuffer = new(std::nothrow) MxpiBuffer {bufferMap_[frameId].mxpiBuffer->buffer};
        if (tmpBuffer == nullptr) {
            return APP_ERR_COMM_ALLOC_MEM;
        }
        SendData(0, *tmpBuffer);
        SendData(1, *bufferMap_[frameId].mxpiBuffer);
    }
    auto bufferMapIter = std::find(bufferMap_[frameId].trackIdVec.begin(),
                                   bufferMap_[frameId].trackIdVec.end(), trackId);
    if (bufferMapIter != bufferMap_[frameId].trackIdVec.end()) {
        bufferMap_[frameId].trackIdVec.erase(bufferMapIter);
        bufferMap_[frameId].ref--;
    }
    return APP_ERR_OK;
}

APP_ERROR MxpiFaceSelection::AddMetaData(MxTools::MxpiBuffer &buffer, std::shared_ptr<MxpiObjectList> &objectList,
                                         std::shared_ptr<MxTools::MxpiKeyPointAndAngleList> &keyPointAndAngleList)
{
    MxpiMetadataManager metadataManager(buffer);
    std::string objectListKey = pluginName_ + "_0";
    APP_ERROR ret = metadataManager.AddProtoMetadata(objectListKey,
                                                     std::static_pointer_cast<void>(objectList));
    if (ret != APP_ERR_OK) {
        errorInfo_ << GetError(ret, pluginName_) << "Fail to add metadata.";
        return ret;
    }
    std::string keyPointAndAngleListKey = pluginName_ + "_1";
    ret = metadataManager.AddProtoMetadata(keyPointAndAngleListKey,
                                           std::static_pointer_cast<void>(keyPointAndAngleList));
    if (ret != APP_ERR_OK) {
        errorInfo_ << GetError(ret, pluginName_) << "Fail to add metadata.";
        return ret;
    }
    return APP_ERR_OK;
}

void MxpiFaceSelection::GetObjectListResult(std::map<uint32_t, FaceObject>::iterator &iter,
                                            std::shared_ptr<MxTools::MxpiObjectList> &objectList)
{
    MxpiObject* object = objectList->add_objectvec();
    MxpiClass* mxpiClass = object->add_classvec();
    MxpiMetaHeader* objectMetaHeader = object->add_headervec();
    object->set_x0(iter->second.detectInfo.x0());
    object->set_x1(iter->second.detectInfo.x1());
    object->set_y0(iter->second.detectInfo.y0());
    object->set_y1(iter->second.detectInfo.y1());
    mxpiClass->set_classid(iter->second.detectInfo.classvec(0).classid());
    mxpiClass->set_classname(iter->second.detectInfo.classvec(0).classname());
    mxpiClass->set_confidence(iter->second.detectInfo.classvec(0).confidence());
    objectMetaHeader->set_datasource(iter->second.parentName);
    objectMetaHeader->set_memberid(iter->second.memberId);
}

void MxpiFaceSelection::GetKeyPointResult(std::map<uint32_t, FaceObject>::iterator &iter,
                                          std::shared_ptr<MxTools::MxpiKeyPointAndAngleList> &keyPointAndAngleList)
{
    MxpiKeyPointAndAngle* keyPointAndAngle = keyPointAndAngleList->add_keypointandanglevec();
    MxpiMetaHeader* keyPointHeader = keyPointAndAngle->add_headervec();
    for (int i = 0; i < iter->second.keyPointAndAngle.keypointsvec_size(); ++i) {
        keyPointAndAngle->add_keypointsvec(iter->second.keyPointAndAngle.keypointsvec(i));
    }
    keyPointAndAngle->set_anglepitch(iter->second.keyPointAndAngle.anglepitch());
    keyPointAndAngle->set_angleroll(iter->second.keyPointAndAngle.angleroll());
    keyPointAndAngle->set_angleyaw(iter->second.keyPointAndAngle.angleyaw());
    keyPointHeader->set_datasource(iter->second.parentName);
    keyPointHeader->set_memberid(iter->second.memberId);
}

namespace {
MX_PLUGIN_GENERATE(MxpiFaceSelection)
}