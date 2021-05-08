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

#ifndef MXPLUGINS_MPOBJECTSELECTION_H
#define MXPLUGINS_MPOBJECTSELECTION_H

#include <stack>
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/buffer/MxpiBufferManager.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"

struct TargetTrack {
    MxTools::MxpiTrackLet mxpiTrackLet;
    MxTools::MxpiVision mxpiVision;
    MxTools::MxpiObject mxpiObject;
    MxBase::MemoryData data;
    uint32_t channelId;
    float score;
    float marginScore;
    float occludeScore;
    float sizeScore;
    float confScore;
    int imageHeight;
    int imageWidth;
    int age;
};

class MpObjectSelection : public MxTools::MxPluginBase {
public:
    /**
    * @api
    * @param configParamMap.
    * @return Error code.
    */
    APP_ERROR Init(std::map<std::string, std::shared_ptr<void>>& configParamMap) override;

    /**
    * @api
    * @return Error code.
    */
    APP_ERROR DeInit() override;

    /**
    * @api
    * @brief Definition the parameter of configure properties.
    * @return Error code.
    */
    APP_ERROR Process(std::vector<MxTools::MxpiBuffer*>& mxpiBuffer) override;

    /**
    * @api
    * @brief Definition the parameter of configure properties.
    * @return std::vector<std::shared_ptr<void>>
    */
    static std::vector<std::shared_ptr<void>> DefineProperties();

    /**
    * @api
    * @brief Define the number and data type of input ports.
    * @return MxTools::MxpiPortInfo.
    */
    static MxTools::MxpiPortInfo DefineInputPorts();

    /**
    * @api
    * @brief Define the number and data type of output ports.
    * @return MxTools::MxpiPortInfo.
    */
    static MxTools::MxpiPortInfo DefineOutputPorts();

private:
    void GetNormRadius(std::string& normRadius);
    APP_ERROR SetMxpiErrorInfo(const std::string pluginName, APP_ERROR errorCode, const std::string& errorText);
    APP_ERROR CheckInputBuffer(MxTools::MxpiBuffer& motBuffer);
    APP_ERROR TargetSelect(MxTools::MxpiBuffer& buffer, std::shared_ptr<MxTools::MxpiTrackLetList>& datalist);
    APP_ERROR GetPositionScore(const MxTools::MxpiObject& mxpiObject, TargetTrack& targetTrack);
    APP_ERROR GetOccludeScore(TargetTrack& targetTrack);
    APP_ERROR PushDataToStack(MxTools::MxpiBuffer& buffer, MxTools::MxpiTrackLetList& mxpiTrackLetList, float yMax);
    APP_ERROR PushObject(float yMax, MxTools::MxpiTrackLetList& trackLetList,
        std::shared_ptr<MxTools::MxpiObjectList>& mxpiObjectList);
    APP_ERROR StartSelect(MxTools::MxpiVisionList& cropVisionList, int imageHeight, int imageWidth);
    APP_ERROR RefleshData(TargetTrack& targetTrack);
    APP_ERROR CreatNewBuffer(const int trackId, bool refresh = true);
    APP_ERROR AddObjectList(MxTools::MxpiBuffer& buffer, std::map<int, TargetTrack>::iterator& iter);
    bool CheckSendData(const int trackId);
private:
    std::string prePluginName_ = "";
    std::string cropPluginName_ = "";
    std::ostringstream errorInfo_ {};
    std::map<int, TargetTrack> targetTrack_ = {};
    std::stack<TargetTrack> stackSet_ = {};
    std::vector<TargetTrack> frontSet_ = {};
    std::vector<float> normRadius_ = {};
    std::vector<std::string> keysVec_ = {};
    uint32_t frameId_ = 0;
    uint32_t channelId_ = 0;
    int trackTime_ = 0;
    float tmargin_ = 0.f;
    float weightMargin_ = 0.f;
    float weightOcclude_ = 0.f;
    float weightSize_ = 0.f;
    float weightConf_ = 0.f;
};
#endif
