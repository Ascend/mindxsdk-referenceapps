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
#include "PluginAlone.h"

using namespace MxPlugins;
using namespace MxTools;
using namespace std;

namespace MxPlugins {
    APP_ERROR PluginAlone::Init(std::map<std::string, std::shared_ptr<void>> &configParamMap) {
        LogInfo << "PluginAlone::Init start.";
        APP_ERROR ret = APP_ERR_OK;
        // Get the property values by key
        // Data from track plugin
        std::shared_ptr<string> tracksourcePropSptr =
                std::static_pointer_cast<string>(configParamMap["dataSourceTrack"]);
        tracksource_ = *tracksourcePropSptr.get();
        std::shared_ptr<string> dataSourceDetection =
                std::static_pointer_cast<string>(configParamMap["dataSourceDetection"]);
        detectionsource_ = *dataSourceDetection.get();
        // Description message
        std::shared_ptr<string> descriptionMessageProSptr =
                std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
        descriptionMessage_ = *descriptionMessageProSptr.get();
        // Configuration parameter
        confthres_ = *std::static_pointer_cast<uint>(configParamMap["detectThresh"]);
        confratio_ = *std::static_pointer_cast<float>(configParamMap["detectRatio"]);
        confsleep_ = *std::static_pointer_cast<uint>(configParamMap["detectSleep"]);

        return APP_ERR_OK;
    }

    APP_ERROR PluginAlone::DeInit() {
        LogInfo << "PluginAlone::DeInit end.";
        return APP_ERR_OK;
    }

    APP_ERROR PluginAlone::SetMxpiErrorInfo(MxpiBuffer &buffer, const std::string pluginName,
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

    APP_ERROR PluginAlone::Process(std::vector<MxpiBuffer *> &mxpiBuffer) {
        LogInfo << "PluginAlone::Process start";
        MxpiBuffer *buffer = mxpiBuffer[0];
        MxpiMetadataManager mxpiMetadataManager(*buffer);
        MxpiErrorInfo mxpiErrorInfo;
        ErrorInfo_.str("");
        auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
        if (errorInfoPtr != nullptr) {
            ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_)
                       << "PluginAlone process is not implemented";
            mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
            mxpiErrorInfo.errorInfo = ErrorInfo_.str();
            SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
            LogError << "PluginAlone process is not implemented";
            return APP_ERR_COMM_FAILURE;
        }
        // Get the data from buffer
        shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(tracksource_);
        shared_ptr<void> Detect = mxpiMetadataManager.GetMetadata(detectionsource_);
        if (metadata == nullptr || Detect == nullptr) {
            ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_)
                       << "Metadata is NULL, failed";
            mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
            mxpiErrorInfo.errorInfo = ErrorInfo_.str();
            SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
            return APP_ERR_METADATA_IS_NULL;
        }
        // get trackletlist
        std::shared_ptr<MxpiTrackLetList> srcTrackLetListSptr = std::static_pointer_cast<MxpiTrackLetList>(metadata);
        std::shared_ptr<MxpiObjectList> srcObjectListSptr = std::static_pointer_cast<MxpiObjectList>(Detect);
        // Data processing
        std::shared_ptr<MxTools::MxpiAttributeList> result = std::make_shared<MxTools::MxpiAttributeList>();
        MxpiAttribute *mxpiAttribute = result->add_attributevec();
        if (sleeptime == 0) {
            LogInfo << "PluginAlone start";
            int alarm = calculate(data_queue, confthres_, confratio_,
                                  srcTrackLetListSptr, srcObjectListSptr);
            if (alarm == 1) {
                LogInfo << "Alarm alone";
                alarm_count++;
                LogInfo << "Alarmed " << alarm_count << " times";
                sleeptime = confsleep_;
                data_queue.clear();
                // Send the data to downstream plugin
                mxpiAttribute->set_attrname("Alarm Alone");
                APP_ERROR ret = mxpiMetadataManager.AddProtoMetadata(pluginName_,
                                                                     static_pointer_cast<void>(result));
                if (ret != APP_ERR_OK) {
                    ErrorInfo_ << GetError(ret, pluginName_) << "PluginAlone add metadata failed.";
                    mxpiErrorInfo.ret = ret;
                    mxpiErrorInfo.errorInfo = ErrorInfo_.str();
                    SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
                    return ret;
                }
            } else {
                mxpiAttribute->set_attrname("No Alarm");
                APP_ERROR ret = mxpiMetadataManager.AddProtoMetadata(pluginName_,
                                                                     static_pointer_cast<void>(result));
                if (ret != APP_ERR_OK) {
                    ErrorInfo_ << GetError(ret, pluginName_) << "PluginAlone add metadata failed.";
                    mxpiErrorInfo.ret = ret;
                    mxpiErrorInfo.errorInfo = ErrorInfo_.str();
                    SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
                    return ret;
                }
            }
        } else {
            mxpiAttribute->set_attrname("Alarmed in a short period of time");
            APP_ERROR ret = mxpiMetadataManager.AddProtoMetadata(pluginName_,
                                                                 static_pointer_cast<void>(result));
            if (ret != APP_ERR_OK) {
                ErrorInfo_ << GetError(ret, pluginName_) << "PluginAlone add metadata failed.";
                mxpiErrorInfo.ret = ret;
                mxpiErrorInfo.errorInfo = ErrorInfo_.str();
                SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
                return ret;
            }
            sleeptime--;
        }
        // Send the data to downstream plugin
        SendData(0, *buffer);
        LogInfo << "PluginAlone::Process end";
        return APP_ERR_OK;
    }

    int PluginAlone::calculate(std::vector<int> &data_queue, int confthres_, float confratio_,
                               std::shared_ptr<MxpiTrackLetList> srcTrackLetListSptr,
                               std::shared_ptr<MxpiObjectList> srcObjectListSptr) {
        uint32_t count = 0;
        uint32_t thresh_ = (int) (confthres_ * confratio_);
        // update data
        for (uint32_t i = 0; i < (uint32_t) srcTrackLetListSptr->trackletvec_size(); i++) {
            auto &trackObject = srcTrackLetListSptr->trackletvec(i);
            if (trackObject.headervec_size() == 0) {
                continue;
            }
            auto &detectObject = srcObjectListSptr->objectvec(trackObject.headervec(0).memberid());
            if (detectObject.classvec(0).classid() != 0) {
                continue;
            }
            count++;
        }
        uint32_t result_present = 0;
        uint32_t result_total = 0;
        if (count == 1) {
            result_present = 1;
        }
        int alarm = 0;
        if (data_queue.size() == confthres_) {
            for (int i = confthres_ - 1; i >= 0; i--) {
                if (data_queue[i] == 1) {
                    result_total++;
                }
                if (i != confthres_ - 1) {
                    data_queue[i + 1] = data_queue[i];
                }
            }
            data_queue[0] = result_present;
            if (result_total >= thresh_) {
                alarm = 1;
            }
        } else {
            data_queue.push_back(result_present);
        }
        return alarm;
    }

    std::vector<std::shared_ptr<void>> PluginAlone::DefineProperties() {
        // Define an A to store properties
        std::vector<std::shared_ptr<void>> properties;
        // Set the type and related information of the properties, and the key is the name
        auto tracksourceProSptr =
                std::make_shared<ElementProperty<string >>(ElementProperty<string>{
                        STRING, "dataSourceTrack", "detect",
                        "the name of previous plugin", "mxpi_motsimplesort0",
                        "NULL", "NULL"});

        auto detectsourceProSptr =
                std::make_shared<ElementProperty<string >>(ElementProperty<string>{
                        STRING, "dataSourceDetection", "name",
                        "the name of previous plugin", "mxpi_objectpostprocessor0",
                        "NULL", "NULL"});

        auto threshProSptr =
                std::make_shared<ElementProperty<uint32_t >>(ElementProperty<uint32_t>{
                        UINT, "detectThresh", "thresh",
                        "the number of frame when judging",
                        100, 0, 1000});

        auto ratioProSptr =
                std::make_shared<ElementProperty<float >>(ElementProperty<float>{
                        FLOAT, "detectRatio", "ratio",
                        "the ratio of judging",
                        0.8, 0.0, 1.0});

        auto sleepProSptr =
                std::make_shared<ElementProperty<uint32_t >>(ElementProperty<uint32_t>{
                        UINT, "detectSleep", "sleep",
                        "the time of stop detection",
                        8, 0, 300});

        auto descriptionMessageProSptr =
                std::make_shared<ElementProperty<string >>(ElementProperty<string>{
                        STRING, "descriptionMessage", "message",
                        "Description mesasge of plugin",
                        "This is PluginAlone", "NULL", "NULL"});
        properties.push_back(tracksourceProSptr);
        properties.push_back(detectsourceProSptr);
        properties.push_back(threshProSptr);
        properties.push_back(ratioProSptr);
        properties.push_back(sleepProSptr);
        properties.push_back(descriptionMessageProSptr);
        return properties;
    }

    // Register the Sample plugin through macro
    MxpiPortInfo PluginAlone::DefineInputPorts() {
        MxpiPortInfo inputPortInfo;
        std::vector<std::vector<std::string>> value = {{"ANY"}};
        MxPluginBase::GenerateStaticInputPortsInfo(value, inputPortInfo);
        return inputPortInfo;
    };
}

namespace {
    MX_PLUGIN_GENERATE(PluginAlone)
}
