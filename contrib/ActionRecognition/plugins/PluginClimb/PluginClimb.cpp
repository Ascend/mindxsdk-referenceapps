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
#include <opencv2/opencv.hpp>
#include "MxBase/Log/Log.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "PluginClimb.h"

using namespace MxPlugins;
using namespace MxTools;
using namespace std;

namespace {
    const int indx_num = 4;
    const int y_indx = 3;
    const int up_indx = 2;
    const int in_indx = 1;
}

namespace MxPlugins {
    APP_ERROR PluginClimb::Init(std::map<std::string, std::shared_ptr<void>> &configParamMap) {
        LogInfo << "PluginClimb::Init start.";
        APP_ERROR ret = APP_ERR_OK;
        // Get the property values by key
        std::shared_ptr<string> tracksourcePropSptr =
                std::static_pointer_cast<string>(configParamMap["dataSourceTrack"]);
        tracksource_ = *tracksourcePropSptr.get();
        LogInfo << "PluginClimb::Init start.";
        std::shared_ptr<string> dataSourceDetection =
                std::static_pointer_cast<string>(configParamMap["dataSourceDetection"]);
        detectionsource_ = *dataSourceDetection.get();
        LogInfo << "PluginClimb::Init start.";
        std::shared_ptr<string> descriptionMessageProSptr =
                std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
        descriptionMessage_ = *descriptionMessageProSptr.get();
        LogInfo << "PluginClimb::Init start.";
        // Configuration parameter
        highthresh_ = *std::static_pointer_cast<int>(configParamMap["highThresh"]);
        LogInfo << "PluginClimb::Init start.";
        bufferlength_ = *std::static_pointer_cast<int>(configParamMap["bufferLength"]);
        LogInfo << "PluginClimb::Init start.";
        ratio_ = *std::static_pointer_cast<float>(configParamMap["detectRatio"]);
        LogInfo << "PluginClimb::Init start.";
        detectsleep_ = *std::static_pointer_cast<int>(configParamMap["detectSleep"]);
        LogInfo << "PluginClimb::Init start.";
        filepath_ = *std::static_pointer_cast<string>(configParamMap["filePath"]);
        LogInfo << "PluginClimb::Init end.";
        return APP_ERR_OK;
    }

    APP_ERROR PluginClimb::DeInit() {
        LogInfo << "PluginClimb::DeInit end.";
        return APP_ERR_OK;
    }

    APP_ERROR PluginClimb::SetMxpiErrorInfo(MxpiBuffer &buffer, const std::string pluginName,
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

    APP_ERROR PluginClimb::Process(std::vector<MxpiBuffer *> &mxpiBuffer) {
        LogInfo << "PluginClimb::Process start";
        MxpiBuffer *buffer = mxpiBuffer[0];
        MxpiMetadataManager mxpiMetadataManager(*buffer);
        MxpiErrorInfo mxpiErrorInfo;
        ErrorInfo_.str("");
        auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
        if (errorInfoPtr != nullptr) {
            ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_)
                       << "PluginClimb process is not implemented";
            mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
            mxpiErrorInfo.errorInfo = ErrorInfo_.str();
            SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
            LogError << "PluginClimb process is not implemented";
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
            return APP_ERR_METADATA_IS_NULL; // self define the error code
        }
        // get trackletlist
        std::shared_ptr<MxpiTrackLetList> srcTrackLetListSptr = std::static_pointer_cast<MxpiTrackLetList>(metadata);
        std::shared_ptr<MxpiObjectList> srcObjectListSptr = std::static_pointer_cast<MxpiObjectList>(Detect);
        // update data
        if (pathflag == 0) {
            readTxt(filepath_, roi);
            pathflag = 1;
        }
        std::shared_ptr<MxTools::MxpiAttributeList> result = std::make_shared<MxTools::MxpiAttributeList>();
        MxpiAttribute *mxpiAttribute = result->add_attributevec();
        if (sleeptime == 0) {
            LogInfo << "PluginClimb start";
            int alarm = calculate(bufferlength_, highthresh_, ratio_, roi, srcTrackLetListSptr,
                                  srcObjectListSptr);
            if (alarm == 1) {
                alarm_count++;
                LogInfo << "Alarmed " << alarm_count << " times";
                trackdata.clear();
                sleeptime = detectsleep_;
                // Send the data to downstream plugin
                mxpiAttribute->set_attrname("Alarm Climb up");
                APP_ERROR ret = mxpiMetadataManager.AddProtoMetadata(pluginName_,
                                                                     static_pointer_cast<void>(result));
                if (ret != APP_ERR_OK) {
                    ErrorInfo_ << GetError(ret, pluginName_) << "PluginClimb add metadata failed.";
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
                    ErrorInfo_ << GetError(ret, pluginName_) << "PluginClimb add metadata failed.";
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
                ErrorInfo_ << GetError(ret, pluginName_) << "PluginClimb add metadata failed.";
                mxpiErrorInfo.ret = ret;
                mxpiErrorInfo.errorInfo = ErrorInfo_.str();
                SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
                return ret;
            }
            sleeptime--;
        }
        LogInfo << "PluginClimb end";
        SendData(0, *buffer);
        LogInfo << "PluginClimb::Process end";
        return APP_ERR_OK;
    }

    int PluginClimb::calculate(int bufferlength_, int highthresh_, float ratio_,
                               vector<cv::Point> roi,
                               std::shared_ptr<MxpiTrackLetList> srcTrackLetListSptr,
                               std::shared_ptr<MxpiObjectList> srcObjectListSptr) {
        int alarm = 0;
        for (uint32_t i = 0; i < (uint32_t) srcTrackLetListSptr->trackletvec_size(); i++) {
            auto &trackObject = srcTrackLetListSptr->trackletvec(i);
            if (trackObject.headervec_size() == 0) {
                continue;
                // As long as the person who has appeared will have a trackobject.
                // Determine whether this id appears in the current frame.
            }
            auto &detectObject = srcObjectListSptr->objectvec(trackObject.headervec(0).memberid());
            if (detectObject.classvec(0).classid() != 0) {
                continue;
            }
            int cnt_x = (detectObject.x0() + detectObject.x1()) / 2;
            int cnt_y = (detectObject.y0() + detectObject.y1()) / 2;
            if (trackdata.count(trackObject.trackid()) == 0) {
                int up_num = 0;
                int in_num = 0;
                if (cv::pointPolygonTest(roi, cv::Point(cnt_x, cnt_y), false) > 0) {
                    in_num += 1;
                }
                trackdata[trackObject.trackid()].push_back(cnt_x); // store cnt_x
                trackdata[trackObject.trackid()].push_back(cnt_y); // store cnt_y
                trackdata[trackObject.trackid()].push_back(up_num); // store up_num
                trackdata[trackObject.trackid()].push_back(in_num); // store in_num
            } else {
                int up_num = trackdata[trackObject.trackid()][trackdata[trackObject.trackid()].size() - up_indx];
                int in_num = trackdata[trackObject.trackid()].back();
                if (cv::pointPolygonTest(roi, cv::Point(cnt_x, cnt_y), false) > 0) {
                    in_num = trackdata[trackObject.trackid()].back() + 1;
                }  // if the target is in the area, in_num ++
                if (cnt_y > trackdata[trackObject.trackid()][trackdata[trackObject.trackid()].size() - y_indx]) {
                    up_num = trackdata[trackObject.trackid()][trackdata[trackObject.trackid()].size() - up_indx] + 1;
                } // if the target is rising, up_num ++
                trackdata[trackObject.trackid()].push_back(cnt_x); // store cnt_x
                trackdata[trackObject.trackid()].push_back(cnt_y); // store cnt_y
                trackdata[trackObject.trackid()].push_back(up_num); // store up_num
                trackdata[trackObject.trackid()].push_back(in_num); // store in_num     
            }
            if (trackdata[trackObject.trackid()].size() == indx_num * bufferlength_) {
                if (trackdata[trackObject.trackid()][indx_num * bufferlength_ - up_indx] >
                    (int) (bufferlength_ * ratio_)
                    && (trackdata[trackObject.trackid()][indx_num * bufferlength_ - in_indx] >
                        (int) (bufferlength_ * ratio_))
                    && trackdata[trackObject.trackid()][indx_num * bufferlength_ - y_indx] -
                       trackdata[trackObject.trackid()][1] > highthresh_) {
                    alarm = 1;
                } else {
                    // Climbing up judgment failed, delete the last frame of the queue, continue to load data
                    std::vector<int> tmp(trackdata[trackObject.trackid()].begin() + indx_num,
                                         trackdata[trackObject.trackid()].end());
                    trackdata[trackObject.trackid()] = tmp;
                }
            }
        }
        return alarm;
    }

    std::vector<std::shared_ptr<void>> PluginClimb::DefineProperties() {
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

        auto highthreshProSptr =
                std::make_shared<ElementProperty<int >>(ElementProperty<int>{
                        UINT, "highThresh", "name",
                        "the name of previous plugin", 8,
                        0, 1000});

        auto bufferlengthProSptr =
                std::make_shared<ElementProperty<int >>(ElementProperty<int>{
                        UINT, "bufferLength", "name",
                        "the name of previous plugin", 10,
                        8, 1000});

        auto ratioProSptr =
                std::make_shared<ElementProperty<float >>(ElementProperty<float>{
                        FLOAT, "detectRatio", "name",
                        "the name of previous plugin", 0.75,
                        0, 1});

        auto sleepProSptr =
                std::make_shared<ElementProperty<int >>(ElementProperty<int>{
                        UINT, "detectSleep", "name", " ", 8, 0,
                        100});

        auto filepathProSptr =
                std::make_shared<ElementProperty<string >>(ElementProperty<string>{
                        STRING, "filePath", "name",
                        "the name of previous plugin", "mxpi_fairmot0",
                        "NULL", "NULL"});

        auto descriptionMessageProSptr =
                std::make_shared<ElementProperty<string >>(ElementProperty<string>{
                        STRING, "descriptionMessage", "message",
                        "Description mesasge of plugin",
                        "This is PluginClimb", "NULL", "NULL"});
        properties.push_back(tracksourceProSptr);
        properties.push_back(detectsourceProSptr);
        properties.push_back(highthreshProSptr);
        properties.push_back(bufferlengthProSptr);
        properties.push_back(ratioProSptr);
        properties.push_back(sleepProSptr);
        properties.push_back(filepathProSptr);
        properties.push_back(descriptionMessageProSptr);
        return properties;
    }

    // Register the Sample plugin through macro
    MxpiPortInfo PluginClimb::DefineInputPorts() {
        MxpiPortInfo inputPortInfo;
        std::vector<std::vector<std::string>> value = {{"ANY"}};
        MxPluginBase::GenerateStaticInputPortsInfo(value, inputPortInfo);
        return inputPortInfo;
    };

    MxpiPortInfo PluginClimb::DefineOutputPorts() {
        MxpiPortInfo outputPortInfo;
        std::vector<std::vector<std::string>> value = {{"ANY"}};
        MxPluginBase::GenerateStaticOutputPortsInfo(value, outputPortInfo);
        return outputPortInfo;
    }
    // log in the txt of roi

    void PluginClimb::readTxt(string file, vector<cv::Point> &roi) {
        ifstream infile;
        infile.open(file);
        string str;
        while (getline(infile, str)) {
            int i = 0;
            cv::Point p;
            int count = 1;
            while (i < str.length()) {
                int j = i;
                while (j < str.length() && str[j] != ';') {
                    j++;
                }
                string s = str.substr(i, j - i);
                if (count == 1) {
                    p.x = atoi(s.c_str());
                    count++;
                } else if (count == 2) {
                    p.y = atoi(s.c_str());
                }
                i = j + 1;
            }
            roi.push_back(p);
        }
        infile.close();
    }
}

namespace {
    MX_PLUGIN_GENERATE(PluginClimb)
}
