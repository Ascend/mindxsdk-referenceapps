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

#ifndef MXPLUGINS__FACESELECTION_H
#define MXPLUGINS__FACESELECTION_H

#include <MxTools/PluginToolkit/buffer/MxpiBufferManager.h>
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/CV/Core/DataType.h"

struct FaceObject {
    std::string parentName;
    uint32_t memberId;
    uint32_t frameId;
    uint32_t channelId;
    float score;
    MxTools::MxpiTrackLet trackLet;
    MxTools::MxpiObject detectInfo;
    MxTools::MxpiKeyPointAndAngle keyPointAndAngle;
};

struct BufferManager {
    int ref;
    MxTools::MxpiBuffer* mxpiBuffer;
    std::vector<uint32_t> trackIdVec;
};

class MxpiFaceSelection : public MxTools::MxPluginBase {
public:
    /**
     * @api
     * @param configParamMap
     * @return
     */
    APP_ERROR Init(std::map<std::string, std::shared_ptr<void>> &configParamMap) override;

    /**
     * @api
     * @return
     */
    APP_ERROR DeInit() override;

    /**
     * @api
     * @param mxpiBuffer
     * @return
     */
    APP_ERROR Process(std::vector<MxTools::MxpiBuffer*> &mxpiBuffer) override;

    /**
     * @api
     * @brief Definition the parameter of configure properties.
     * @return std::vector<std::shared_ptr<void>>
     */
    static std::vector<std::shared_ptr<void>> DefineProperties();

    /**
     * Optional, defines input ports of the plugin.
     */
    static MxTools::MxpiPortInfo DefineInputPorts();

    /**
     * Optional, defines output ports of the plugin.
     */
    static MxTools::MxpiPortInfo DefineOutputPorts();

private:
    APP_ERROR GetPrePluginsResult(MxTools::MxpiBuffer &inputBuffer, std::vector<FaceObject> &faceObjectQueue);

    float CalKeyPointScore(const FaceObject &faceObject);

    float CalEulerScore(const FaceObject &faceObject);

    float CalFaceSizeScore(const FaceObject &faceObject);

    float CalTotalScore(const FaceObject &faceObject);

    void FaceQualityEvaluation(std::vector<FaceObject> &faceObjectQueue, MxTools::MxpiBuffer &buffer);

    APP_ERROR GetFaceSelectionResult();

    APP_ERROR SendSelectionDate(std::map<uint32_t, FaceObject>::iterator &iter,
                                std::shared_ptr<MxTools::MxpiObjectList> &objectList,
                                std::shared_ptr<MxTools::MxpiKeyPointAndAngleList> &keyPointAndAngleList);

    APP_ERROR AddMetaData(MxTools::MxpiBuffer &buffer, std::shared_ptr<MxTools::MxpiObjectList> &objectList,
                          std::shared_ptr<MxTools::MxpiKeyPointAndAngleList> &keyPointAndAngleList);

    void GetObjectListResult(std::map<uint32_t, FaceObject>::iterator &iter,
                             std::shared_ptr<MxTools::MxpiObjectList> &objectList);

    void GetKeyPointResult(std::map<uint32_t, FaceObject>::iterator &iter,
                           std::shared_ptr<MxTools::MxpiKeyPointAndAngleList> &keyPointAndAngleList);

    APP_ERROR ErrorProcess(MxTools::MxpiBuffer &inputBuffer);

    APP_ERROR CheckMetadataType(MxTools::MxpiBuffer &inputBuffer);

private:
    float keyPointWeight_ = 0.f; // weight of key point score
    float eulerWeight_ = 0.f;  // weight of face euler angles score
    float faceSizeWeight_ = 0.f; // weight of face score
    float minScoreThreshold_ = 0.f; // min face total score threshold
    uint32_t maxAge_ = 0; // max age for stopping face selection
    std::string trackedParentName_ = ""; // the key of trackLet input data
    std::string keyPointParentName_ = ""; // the key of face key point input data
    std::map<uint32_t, FaceObject> qualityAssessmentMap_ = {};
    std::map<uint32_t, BufferManager> bufferMap_ = {};
    bool isUsedBuffer_ = false; // if is used current input buffer
    std::ostringstream errorInfo_ {}; // the error information
};

#endif
