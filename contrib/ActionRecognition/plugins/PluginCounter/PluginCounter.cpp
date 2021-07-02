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

#include "math.h"
#include "MxBase/Log/Log.h"
#include "MxTools/Proto/MxpiDumpData.pb.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "PluginCounter.h"

using namespace MxPlugins;
using namespace MxTools;
using namespace std;

namespace {
    const string SAMPLE_KEY = "MxpiObjectList";
}

namespace MxPlugins {
    APP_ERROR PluginCounter::Init(std::map<std::string, std::shared_ptr<void>> &configParamMap) {
        LogInfo << "PluginCounter::Init start.";
        APP_ERROR ret = APP_ERR_OK;

        // Get the property values by key
        std::shared_ptr<string> tracksourcePropSptr =
                std::static_pointer_cast<string>(configParamMap["dataSourceTrack"]);
        tracksource_ = *tracksourcePropSptr.get();

        std::shared_ptr<string> descriptionMessageProSptr =
                std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
        descriptionMessage_ = *descriptionMessageProSptr.get();

        return APP_ERR_OK;
    }

    APP_ERROR PluginCounter::DeInit() {
        LogInfo << "PluginCounter::DeInit end.";
        return APP_ERR_OK;
    }

    APP_ERROR PluginCounter::SetMxpiErrorInfo(MxpiBuffer &buffer, const std::string pluginName,
                                              const MxpiErrorInfo mxpiErrorInfo) {
        APP_ERROR ret = APP_ERR_OK;
        // Define an object of MxpiMetadataManager
        MxpiMetadataManager mxpiMetadataManager(buffer);
        ret = mxpiMetadataManager.AddErrorInfo(pluginName, mxpiErrorInfo);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to AddErrorInfo.";
            return ret;
        }
        ret = SendData(0, buffer);
        return ret;
    }

    APP_ERROR PluginCounter::Process(std::vector<MxpiBuffer *> &mxpiBuffer) {
        LogInfo << "PluginCounter::Process start";
        MxpiBuffer *buffer = mxpiBuffer[0];

        MxpiMetadataManager mxpiMetadataManager(*buffer);
        MxpiErrorInfo mxpiErrorInfo;
        ErrorInfo_.str("");
        auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
        frame++;
        LogInfo << "当前帧号 :" << frame;
        if (errorInfoPtr != nullptr) {
            ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) << "PluginCounter process is not implemented";
            mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
            mxpiErrorInfo.errorInfo = ErrorInfo_.str();
            SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
            LogError << "PluginCounter process is not implemented";
            return APP_ERR_COMM_FAILURE;
        }
        // Get the data from buffer
        shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(tracksource_);

        if (metadata == nullptr) {
            ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "Metadata is NULL, failed";
            mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
            mxpiErrorInfo.errorInfo = ErrorInfo_.str();
            SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
            return APP_ERR_METADATA_IS_NULL; // self define the error code
        }
        // check whether the proto struct name is MxpiObjectList
        google::protobuf::Message *msg = (google::protobuf::Message *) metadata.get();

        // get trackletlist
        std::shared_ptr<MxpiTrackLetList> srcTrackLetListSptr = std::static_pointer_cast<MxpiTrackLetList>(metadata);
        // Send the data to downstream plugin
        std::shared_ptr<MxTools::MxpiAttribute> result = std::make_shared<MxTools::MxpiAttribute>();
        result->set_attrname("PluginCounter");
        APP_ERROR ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, result);

        if (ret != APP_ERR_OK) {
            ErrorInfo_ << GetError(ret, pluginName_) << "PluginCounter add metadata failed.";
            mxpiErrorInfo.ret = ret;
            mxpiErrorInfo.errorInfo = ErrorInfo_.str();
            SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
            return ret;
        }
        // Send the data to downstream plugin
        SendData(0, *buffer);
        LogInfo << "PluginCounter::Process end";
        return APP_ERR_OK;
    }

    std::vector<std::shared_ptr<void>> PluginCounter::DefineProperties() {
        // Define an A to store properties
        std::vector<std::shared_ptr<void>> properties;
        // Set the type and related information of the properties, and the key is the name
        auto tracksourceProSptr =
                std::make_shared<ElementProperty<string >>(ElementProperty<string>{
                        STRING, "dataSourceTrack", "name",
                        "the name of previous plugin", "mxpi_motsimplesort0", "NULL",
                        "NULL"});

        auto descriptionMessageProSptr =
                std::make_shared<ElementProperty<string >>(ElementProperty<string>{
                        STRING, "descriptionMessage", "message",
                        "Description mesasge of plugin",
                        "This is PluginCounter", "NULL", "NULL"});

        properties.push_back(tracksourceProSptr);
        properties.push_back(descriptionMessageProSptr);
        return properties;
    }
    // Register the Sample plugin through macro
    MxpiPortInfo PluginCounter::DefineInputPorts() {
        MxpiPortInfo inputPortInfo;
        std::vector<std::vector<std::string>> value = {{"ANY"}};
        MxPluginBase::GenerateStaticInputPortsInfo(value, inputPortInfo);
        return inputPortInfo;
    };

    MxpiPortInfo PluginCounter::DefineOutputPorts() {
        MxpiPortInfo outputPortInfo;
        std::vector<std::vector<std::string>> value = {{"ANY"}};
        MxPluginBase::GenerateStaticOutputPortsInfo(value, outputPortInfo);
        return outputPortInfo;
    }
}
namespace {
    MX_PLUGIN_GENERATE(PluginCounter)
}
