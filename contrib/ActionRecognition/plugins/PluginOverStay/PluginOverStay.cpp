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
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "PluginOverStay.h"

using namespace MxPlugins;
using namespace MxTools;
using namespace std;

namespace MxPlugins {
    APP_ERROR PluginOverStay::Init(std::map<std::string, std::shared_ptr<void>> &configParamMap) {
        LogInfo << "PluginOverStay::Init start.";
        APP_ERROR ret = APP_ERR_OK;
        // Get the property values by key
        std::shared_ptr<string> tracksourcePropSptr =
                std::static_pointer_cast<string>(configParamMap["dataSourceTrack"]);
        tracksource_ = *tracksourcePropSptr.get();

        std::shared_ptr<string> dataSourceDetection =
                std::static_pointer_cast<string>(configParamMap["dataSourceDetection"]);
        detectionsource_ = *dataSourceDetection.get();

        std::shared_ptr<string> descriptionMessageProSptr =
                std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
        descriptionMessage_ = *descriptionMessageProSptr.get();

        // Configuration parameter
        confthresh_ = *std::static_pointer_cast<int>(configParamMap["stayThresh"]);
        confframes_ = *std::static_pointer_cast<int>(configParamMap["frames"]);
        confsleep_ = *std::static_pointer_cast<int>(configParamMap["detectSleep"]);
        confdistance_ = *std::static_pointer_cast<int>(configParamMap["distanceThresh"]);
        confratio_ = *std::static_pointer_cast<float>(configParamMap["detectRatio"]);
        return APP_ERR_OK;
    }

    APP_ERROR PluginOverStay::DeInit() {
        LogInfo << "PluginOverStay::DeInit end.";
        return APP_ERR_OK;
    }

    APP_ERROR PluginOverStay::SetMxpiErrorInfo(MxpiBuffer &buffer, const std::string pluginName,
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

    APP_ERROR PluginOverStay::Process(std::vector<MxpiBuffer *> &mxpiBuffer) {
        LogInfo << "PluginOverStay::Process start";
        MxpiBuffer *buffer = mxpiBuffer[0];
        MxpiMetadataManager mxpiMetadataManager(*buffer);
        MxpiErrorInfo mxpiErrorInfo;
        ErrorInfo_.str("");
        auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
        frame++;
        if (errorInfoPtr != nullptr) {
            ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_)
                       << "PluginOverStay process is not implemented";
            mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
            mxpiErrorInfo.errorInfo = ErrorInfo_.str();
            SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
            LogError << "PluginOverStay process is not implemented";
            return APP_ERR_COMM_FAILURE;
        }
        // Get the data from buffer
        shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(tracksource_);
        shared_ptr<void> Detect = mxpiMetadataManager.GetMetadata(detectionsource_);

        if (metadata == nullptr) {
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
        // data processing
        std::shared_ptr<MxTools::MxpiAttributeList> result = std::make_shared<MxTools::MxpiAttributeList>();
        MxpiAttribute *mxpiAttribute = result->add_attributevec();
        if (sleeptime == 0) {
            // update data
            int alarm = calculate(trackdata, confframes_, frame_num, confthresh_, confratio_, confdistance_,
                                  srcTrackLetListSptr, srcObjectListSptr);
            if (alarm == 1) {
                alarm_count++;
                LogInfo << "Alarmed " << alarm_count << " times";
                sleeptime = confsleep_;
                mxpiAttribute->set_attrname("Alarm Overstay");
                APP_ERROR ret = mxpiMetadataManager.AddProtoMetadata(pluginName_,
                                                                     static_pointer_cast<void>(result));
                if (ret != APP_ERR_OK) {
                    ErrorInfo_ << GetError(ret, pluginName_) << "PluginOverstay add metadata failed.";
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
                    ErrorInfo_ << GetError(ret, pluginName_) << "PluginOverstay add metadata failed.";
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
                ErrorInfo_ << GetError(ret, pluginName_) << "PluginOverstay add metadata failed.";
                mxpiErrorInfo.ret = ret;
                mxpiErrorInfo.errorInfo = ErrorInfo_.str();
                SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
                return ret;
            }
            sleeptime--;
        }
        // Send the data to downstream plugin
        SendData(0, *buffer);
        LogInfo << "PluginOverStay::Process end";
        return APP_ERR_OK;
    }

    int PluginOverStay::calculate(std::map<int, std::vector<int>> &trackdata, int confframes_, int &frame_num,
                                  int confthresh_, int confratio_, int confdistance_,
                                  std::shared_ptr<MxpiTrackLetList> srcTrackLetListSptr,
                                  std::shared_ptr<MxpiObjectList> srcObjectListSptr) {
        int alarm = 0;
        for (uint32_t i = 0; i < (uint32_t) srcTrackLetListSptr->trackletvec_size(); i++) {
            auto &trackObject = srcTrackLetListSptr->trackletvec(i);
            if (trackObject.headervec_size() == 0) {
                continue;
            }
            auto &detectObject = srcObjectListSptr->objectvec(trackObject.headervec(0).memberid());
            if (detectObject.classvec(0).classid() != 0) {
                continue;
            }
            if (trackdata.count(trackObject.trackid()) == 0) {
                trackdata[trackObject.trackid()].push_back((int) ((detectObject.x0() + detectObject.x1()) / 2));
                trackdata[trackObject.trackid()].push_back((int) ((detectObject.y0() + detectObject.y1()) / 2));
                trackdata[trackObject.trackid()][2] = 1;
            } else {
                int dis = distance(trackdata[trackObject.trackid()][0],
                                   trackdata[trackObject.trackid()][1],
                                   (int) ((detectObject.x0() + detectObject.x1()) / 2),
                                   (int) ((detectObject.y0() + detectObject.y1()) / 2));
                if (dis < confdistance_) {
                    trackdata[trackObject.trackid()][2]++;
                } else {
                    trackdata[trackObject.trackid()][2] = 1;
                }
                trackdata[trackObject.trackid()][0] = (int) ((detectObject.x0() + detectObject.x1()) / 2);
                trackdata[trackObject.trackid()][1] = (int) ((detectObject.y0() + detectObject.y1()) / 2);
            }
        }
        frame_num++;
        // calculate data
        if (frame_num == confframes_) {
            frame_num = 0;
            for (auto iter = trackdata.begin(); iter != trackdata.end(); iter++) {
                if (trackdata[iter->first][2] >= (int) (confratio_ * confthresh_)) {
                    alarm = 1;
                    break;
                }
            }
            trackdata.clear();
        }
        return alarm;
    }

    std::vector<std::shared_ptr<void>> PluginOverStay::DefineProperties() {
        // Define an A to store properties
        std::vector<std::shared_ptr<void>> properties;
        // Set the type and related information of the properties, and the key is the name
        auto tracksourceProSptr =
                std::make_shared<ElementProperty<string >>(ElementProperty<string>{
                        STRING, "dataSourceTrack", "name",
                        "the name of previous plugin", "mxpi_motsimplesort0", "NULL",
                        "NULL"});

        auto detectsourceProSptr =
                std::make_shared<ElementProperty<string >>(ElementProperty<string>{
                        STRING, "dataSourceDetection", "name",
                        "the name of previous plugin", "mxpi_fairmot0",
                        "NULL", "NULL"});

        auto threshProSptr = // 逗留时间阈值
                std::make_shared<ElementProperty<int >>(ElementProperty<int>{
                        UINT, "stayThresh", "name",
                        "the name of previous plugin", 10,
                        10, 1000});

        auto distanceProSptr = // 移动距离阈值
                std::make_shared<ElementProperty<int >>(ElementProperty<int>{
                        UINT, "distanceThresh", "name",
                        "the name of previous plugin", 6,
                        0, 100});

        auto frameProSptr = // 多少帧检测一次
                std::make_shared<ElementProperty<int >>(ElementProperty<int>{
                        UINT, "frames", "name",
                        "the name of previous plugin", 8,
                        8, 100});

        auto sleepProSptr = // 检测后休止时间
                std::make_shared<ElementProperty<int >>(ElementProperty<int>{
                        UINT, "detectSleep", "name", " ", 8, 0,
                        100});

        auto ratioProSptr = // 检出比例
                std::make_shared<ElementProperty<float >>(ElementProperty<float>{
                        FLOAT, "detectRatio", "name", " ", 0.8, 0.1,
                        1.0});

        auto descriptionMessageProSptr =
                std::make_shared<ElementProperty<string >>(ElementProperty<string>{
                        STRING, "descriptionMessage", "message",
                        "Description mesasge of plugin",
                        "This is PluginOverStay", "NULL", "NULL"});
        properties.push_back(tracksourceProSptr);
        properties.push_back(detectsourceProSptr);
        properties.push_back(threshProSptr);
        properties.push_back(frameProSptr);
        properties.push_back(sleepProSptr);
        properties.push_back(distanceProSptr);
        properties.push_back(ratioProSptr);
        properties.push_back(descriptionMessageProSptr);
        return properties;
    }

    MxpiPortInfo PluginOverStay::DefineInputPorts() {
        MxpiPortInfo inputPortInfo;
        std::vector<std::vector<std::string>> value = {{"ANY"}};
        MxPluginBase::GenerateStaticInputPortsInfo(value, inputPortInfo);
        return inputPortInfo;
    };

    MxpiPortInfo PluginOverStay::DefineOutputPorts() {
        MxpiPortInfo outputPortInfo;
        std::vector<std::vector<std::string>> value = {{"ANY"}};
        MxPluginBase::GenerateStaticOutputPortsInfo(value, outputPortInfo);
        return outputPortInfo;
    }

    int PluginOverStay::distance(int x0, int y0, int x1, int y1) {
        int distance = int(sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)));
        return distance;
    }
}
namespace {
    MX_PLUGIN_GENERATE(PluginOverStay)
}
