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

#include <opencv2/opencv.hpp>
#include "MxBase/Log/Log.h"
#include "PluginOutOfBed.h"

using namespace MxPlugins;
using namespace MxTools;
using namespace std;

namespace MxPlugins {
    APP_ERROR PluginOutOfBed::Init(std::map<std::string, std::shared_ptr<void>> &configParamMap) {
        LogInfo << "PluginOutOfBed::Init start.";
        APP_ERROR ret = APP_ERR_OK;
        // Get the property values by key
        // Data from track plugin
        std::shared_ptr<string> tracksourcePropSptr =
                std::static_pointer_cast<string>(configParamMap["dataSourceTrack"]);
        tracksource_ = *tracksourcePropSptr.get();
        // Data from detect plugin
        std::shared_ptr<string> detectsourcePropSptr =
                std::static_pointer_cast<string>(configParamMap["dataSourceDetection"]);
        detectionsource_ = *detectsourcePropSptr.get();
        // Description message
        std::shared_ptr<string> descriptionMessageProSptr =
                std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
        descriptionMessage_ = *descriptionMessageProSptr.get();
        // Configuration parameter
        confthres_ = *std::static_pointer_cast<uint>(configParamMap["detectThresh"]);
        confratio_ = *std::static_pointer_cast<float>(configParamMap["detectRatio"]);
        confsleep_ = *std::static_pointer_cast<uint>(configParamMap["detectSleep"]);
        configpath = *std::static_pointer_cast<string>(configParamMap["configPath"]);
        return APP_ERR_OK;
    }

    APP_ERROR PluginOutOfBed::DeInit() {
        LogInfo << "PluginOutOfBed::DeInit end.";
        return APP_ERR_OK;
    }

    APP_ERROR PluginOutOfBed::SetMxpiErrorInfo(MxpiBuffer &buffer, const std::string pluginName,
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

    APP_ERROR PluginOutOfBed::Process(std::vector<MxTools::MxpiBuffer *> &mxpiBuffer) {
        MxpiBuffer *buffer = mxpiBuffer[0];
        MxpiMetadataManager mxpiMetadataManager(*buffer);
        MxpiErrorInfo mxpiErrorInfo;
        ErrorInfo_.str("");
        auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
        framesnum++;
        if (errorInfoPtr != nullptr) {
            ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) << "PluginOutOfBed process is not implemented";
            mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
            mxpiErrorInfo.errorInfo = ErrorInfo_.str();
            SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
            LogError << "FairMOT_Climb process is not implemented";
            return APP_ERR_COMM_FAILURE;
        }
        // Get the data from buffer
        shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(tracksource_);
        shared_ptr<void> Detect = mxpiMetadataManager.GetMetadata(detectionsource_);
        if (metadata == nullptr) {
            ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "Metadata is NULL, failed";
            mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
            mxpiErrorInfo.errorInfo = ErrorInfo_.str();
            SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
            return APP_ERR_METADATA_IS_NULL; // self define the error code
        }
        // get trackletlist
        std::shared_ptr<MxpiTrackLetList> srcTrackLetListSptr = std::static_pointer_cast<MxpiTrackLetList>(metadata);
        std::shared_ptr<MxpiObjectList> srcObjectListSptr = std::static_pointer_cast<MxpiObjectList>(Detect);
        bool alarm = OutOfBedProcess(srcTrackLetListSptr, srcObjectListSptr);
        LogInfo << "frame:" << framesnum;
        std::shared_ptr<MxpiAttributeList> mxpiAttributeList = std::make_shared<MxpiAttributeList>();
        MxTools::MxpiAttribute *result = mxpiAttributeList->add_attributevec();
        if (alarm) {
            result->set_attrvalue("Alarm Outofbed");

        } else {
            result->set_attrvalue("No Alarm");
        }
        APP_ERROR ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(mxpiAttributeList));
        if (ret != APP_ERR_OK) {
            ErrorInfo_ << GetError(ret, pluginName_) << "PluginOutOfBed add metadata failed.";
            mxpiErrorInfo.ret = ret;
            mxpiErrorInfo.errorInfo = ErrorInfo_.str();
            SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
            return ret;
        }
        LogInfo << "PluginOutOfBed end";
        // Send the data to downstream plugin
        MxTools::MxPluginBase::SendData(0, *buffer);
        return APP_ERR_OK;
    }

    std::vector<std::shared_ptr<void>> PluginOutOfBed::DefineProperties() {
        // Define an A to store properties
        std::vector<std::shared_ptr<void>> properties;
        // Set the type and related information of the properties, and the key is the name
        auto tracksourceProSptr =
                std::make_shared<ElementProperty<string>>
                        (ElementProperty<string>{
                                STRING, "dataSourceTrack", "name",
                                "the name of previous plugin",
                                "mxpi_motsimplesort0", "NULL",
                                "NULL"});
        auto detectsourceProSptr =
                std::make_shared<ElementProperty<string>>
                        (ElementProperty<string>{
                                STRING, "dataSourceDetection", "name",
                                "the name of previous plugin",
                                "mxpi_motsimplesort0", "NULL",
                                "NULL"});
        auto threshProSptr =
                std::make_shared<ElementProperty<int>>
                        (ElementProperty<int>{
                                UINT, "detectThresh", "name", "123", 8, 0,
                                100});
        auto ratioProSptr =
                std::make_shared<ElementProperty<float>>
                        (ElementProperty<float>{
                                FLOAT, "detectRatio", "name", " ", 1.0, 0.0,
                                1.0});
        auto sleepProSptr =
                std::make_shared<ElementProperty<int>>
                        (ElementProperty<int>{
                                UINT, "detectSleep", "name", " ", 8, 0,
                                100});
        auto descriptionMessageProSptr =
                std::make_shared<ElementProperty<string>>
                        (ElementProperty<string>{
                                STRING, "descriptionMessage", "message",
                                "Description mesasge of plugin",
                                "This is FairMOT_Alone", "NULL", "NULL"});
        auto configPathSptr =
                std::make_shared<ElementProperty<string>>
                        (ElementProperty<string>{
                                STRING, "configPath", "message",
                                "Description mesasge of plugin",
                                "This is FairMOT_Alone", "NULL", "NULL"});
        properties.push_back(tracksourceProSptr);
        properties.push_back(detectsourceProSptr);
        properties.push_back(threshProSptr);
        properties.push_back(ratioProSptr);
        properties.push_back(sleepProSptr);
        properties.push_back(configPathSptr);
        properties.push_back(descriptionMessageProSptr);
        return properties;
    }

    MxpiPortInfo PluginOutOfBed::DefineInputPorts() {
        MxpiPortInfo inputPortInfo;
        std::vector<std::vector<std::string>> value = {{"ANY"}};
        MxPluginBase::GenerateStaticInputPortsInfo(value, inputPortInfo);
        return inputPortInfo;
    }

    // Process the txt and get the location of the bed
    void PluginOutOfBed::readTxt(std::string file, std::vector<cv::Point> &roi) {
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

    // out of bed
    bool PluginOutOfBed::OutOfBed(std::vector<int> queue) {
        int length = queue.size();
        int count_H = 0, count_T = 0;
        for (int i = 0; i < length / 2; i++) {
            if (queue[i]) {
                count_H++;
            }
        }
        for (int j = length - 1; j >= length / 2; j--) {
            if (!queue[j]) {
                count_T++;
            }
        }

        if (count_H > length * confratio_ && count_T > length * confratio_) {
            return 1;
        }
        return 0;
    }

    bool PluginOutOfBed::OutOfBedProcess(std::shared_ptr<MxpiTrackLetList> srcTrackLetListSptr,
                                         std::shared_ptr<MxpiObjectList> srcObjectListSptr) {
        bool flag = false;
        if (sleeptime == 0) {
            cv::Point2f center;
            // read txt and get location of bed
            if (pathflag) {
                readTxt(configpath, bed);
                pathflag = false;
            }
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
                    center.x = (detectObject.x0() + detectObject.x1()) / 2;
                    center.y = (detectObject.y0() + detectObject.y1()) / 2;
                    trackdata[trackObject.trackid()].push_back(cv::pointPolygonTest(bed, center, false) > 0);
                } else {
                    center.x = (detectObject.x0() + detectObject.x1()) / 2;
                    center.y = (detectObject.y0() + detectObject.y1()) / 2;
                    trackdata[trackObject.trackid()].push_back(cv::pointPolygonTest(bed, center, false) > 0);
                    if (trackdata[trackObject.trackid()].size() >= confthres_) {
                        if (OutOfBed(trackdata[trackObject.trackid()])) {
                            LogInfo << "Alarm OutOfBed";
                            alarmcount++;
                            LogInfo << "Alarmed" << alarmcount << "times";
                            flag = true;
                            sleeptime = confsleep_;
                            frames = 0;
                            trackdata.clear();
                            break;
                        } else {
                            trackdata[trackObject.trackid()].erase(trackdata[trackObject.trackid()].begin());
                        }
                    }
                }
            }
            frames++;
        } else {
            LogInfo << "Alarmed in a short period of time";
            sleeptime--;
        }
        return flag;
    }

}
namespace {
    MX_PLUGIN_GENERATE(PluginOutOfBed)
}
