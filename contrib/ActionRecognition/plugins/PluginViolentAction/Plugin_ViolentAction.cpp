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

#include "MxBase/Log/Log.h"
#include "Plugin_ViolentAction.h"

using namespace MxPlugins;
using namespace MxTools;
using namespace MxBase;
using namespace std;

APP_ERROR PluginViolentAction::Init(std::map<std::string, std::shared_ptr<void>> &configParamMap) {
    LogInfo << "Begin to initialize PluginViolentAction(" << pluginName_ << ").";
    // Get the property values by key
    std::shared_ptr<std::string> classSource = std::static_pointer_cast<std::string>(configParamMap["classSource"]);
    classSource_ = *classSource;

    std::shared_ptr<std::string> filePath = std::static_pointer_cast<std::string>(configParamMap["filePath"]);
    filePath_ = *filePath;

    std::shared_ptr<std::uint32_t> detectSleep = std::static_pointer_cast<uint32_t>(configParamMap["detectSleep"]);
    detectSleep_ = *detectSleep;

    std::shared_ptr<std::float_t> actionThreshold = std::static_pointer_cast<float_t>(
            configParamMap["actionThreshold"]);
    actionThreshold_ = *actionThreshold;
    LogInfo << "End to initialize PluginViolentAction(" << pluginName_ << ").";
    return APP_ERR_OK;
}

APP_ERROR PluginViolentAction::DeInit() {
    LogInfo << "Begin to deinitialize PluginViolentAction(" << pluginName_ << ").";
    LogInfo << "End to deinitialize PluginViolentAction(" << pluginName_ << ").";
    return APP_ERR_OK;
}

APP_ERROR PluginViolentAction::CheckDataSource(MxTools::MxpiMetadataManager &mxpiMetadataManager) {
    if (mxpiMetadataManager.GetMetadata(classSource_) == nullptr) {
        LogDebug << GetError(APP_ERR_METADATA_IS_NULL, pluginName_)
        << "class metadata is null. please check"
                 << "Your property classSource(" << classSource_ << ").";
        return APP_ERR_METADATA_IS_NULL;
    }
    return APP_ERR_OK;
}

std::shared_ptr<MxpiAttributeList>
PluginViolentAction::ActionMatch(std::shared_ptr<MxpiClassList> &mxpiClassList) {
    // set MxpiAttributeList
    MxpiClass mxpiClass = mxpiClassList->classvec(0);
    int32_t classId = mxpiClass.classid();
    std::string className = mxpiClass.classname();
    float confidence = mxpiClass.confidence();
    std::shared_ptr<MxpiAttributeList> mxpiAttributeList = std::make_shared<MxpiAttributeList>();
    MxpiAttribute *mxpiAttribute = mxpiAttributeList->add_attributevec();
    mxpiAttribute->set_confidence(confidence);
    mxpiAttribute->set_attrid(classId);
    mxpiAttribute->set_attrname(className);
    if (sleepTime_ == 0) {
        auto iterator = find(aoi.begin(), aoi.end(), className);
        if (iterator != aoi.end()) {
            // mark the times of alarms for statistics
            if (confidence < actionThreshold_) {
                // confidence is too low
                alarmInformation = "Low confidence";
                mxpiAttribute->set_attrvalue(alarmInformation);
                return mxpiAttributeList;
            } else {
                alarm_count++;
                sleepTime_ = detectSleep_;
                alarmInformation = "Alarm Violent Action";
                mxpiAttribute->set_attrvalue(alarmInformation);
                return mxpiAttributeList;
            }
        } else {
            // no interested action
            alarmInformation = "No Alarm";
            mxpiAttribute->set_attrvalue(alarmInformation);
            return mxpiAttributeList;
        }
    } else {
        // alarm sleep
        alarmInformation = "Alarmed in a short period of time";
        mxpiAttribute->set_attrvalue(alarmInformation);
        sleepTime_--;
        return mxpiAttributeList;
    }
}

void PluginViolentAction::ReadTxt(std::string file, std::vector<std::string> &aoi) {
    // read txt file
    std::ifstream infile;
    infile.open(file.data());
    assert(infile.is_open());
    std::string str;
    while (getline(infile, str)) {
        // Remove the ending symbol
        str.erase(str.end() - 1);
        aoi.emplace_back(str);
    }
    infile.close();
}

APP_ERROR PluginViolentAction::Process(std::vector<MxpiBuffer *> &mxpiBuffer) {
    LogInfo << "Begin to process PluginViolentAction(" << elementName_ << ").";
    // Get MxpiClassList from MxpiBuffer
    MxpiBuffer *inputMxpiBuffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*inputMxpiBuffer);
    ErrorInfo_.str("");
    // check data source
    APP_ERROR ret = CheckDataSource(mxpiMetadataManager);
    if (ret != APP_ERR_OK) {
        SendData(0, *inputMxpiBuffer);
        return ret;
    }
    // Get metadata by key
    std::shared_ptr<void> class_metadata = mxpiMetadataManager.GetMetadata(classSource_);
    std::shared_ptr<MxpiClassList> srcClassListPtr = std::static_pointer_cast<MxpiClassList>(class_metadata);
    // update data; Read the action of interest file
    // set pathflag to make file IO only once
    if (pathflag == 0) {
        ReadTxt(filePath_, aoi);
        pathflag = 1;
    }
    // Match the upstream action to the action of interest and alarm.
    auto attributeListPtr = ActionMatch(srcClassListPtr);
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(attributeListPtr));
    if (ret != APP_ERR_OK) {
        LogError << ErrorInfo_.str();
        SendMxpiErrorInfo(*inputMxpiBuffer, pluginName_, ret, ErrorInfo_.str());
        SendData(0, *inputMxpiBuffer);
    }
    // Send the data to downstream plugin
    SendData(0, *inputMxpiBuffer);
    LogInfo << "End to process PluginViolentAction(" << elementName_ << ").";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> PluginViolentAction::DefineProperties() {
    std::vector<std::shared_ptr<void>> properties;
    // Get the action category from previous plugin
    auto classsource = std::make_shared<ElementProperty<std::string>>(ElementProperty<std::string>{
            STRING,
            "classSource",
            "labelSource",
            "Recognized Action Class",
            "default", "NULL", "NULL"
    });
    // The action of interest file path
    auto filepath = std::make_shared<ElementProperty<std::string>>(ElementProperty<std::string>{
            STRING,
            "filePath",
            "Action of interest file path",
            "the path of predefined violent action classes file",
            "NULL", "NULL", "NULL"
    });
    // sleep time after alarm
    auto detectsleep = std::make_shared<ElementProperty<uint>>(ElementProperty<uint>{
            UINT,
            "detectSleep",
            "process sleep time",
            "sleep some time to avoid frequent alarms ",
            8, 0, 100
    });
    // threshold
    auto actionthreshold = std::make_shared<ElementProperty<float>>(ElementProperty<float>{
            FLOAT,
            "actionThreshold",
            "actionThreshold",
            "filter low threshold action",
            0.3, 0.0, 1.0
    });
    properties.push_back(classsource);
    properties.push_back(filepath);
    properties.push_back(detectsleep);
    properties.push_back(actionthreshold);
    return properties;
}

MxpiPortInfo PluginViolentAction::DefineInputPorts() {
    MxpiPortInfo inputPortInfo;
    // Input: {{MxpiClassList}}
    std::vector<std::vector<std::string>> value = {{"ANY"}};
    GenerateStaticInputPortsInfo(value, inputPortInfo);
    return inputPortInfo;
}

MxpiPortInfo PluginViolentAction::DefineOutputPorts() {
    MxpiPortInfo outputPortInfo;
    // Output: {{MxpiAttributeList}}
    std::vector<std::vector<std::string>> value = {{"ANY"}};
    GenerateStaticOutputPortsInfo(value, outputPortInfo);
    return outputPortInfo;
}

namespace {
    MX_PLUGIN_GENERATE(PluginViolentAction)
}
