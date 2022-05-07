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

#include "PassengerFlowEstimation.h"
#include "MxBase/Log/Log.h"
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
namespace {
    const string SAMPLE_KEY = "MxpiObjectList";
    const string SAMPLE_KEY2 = "MxpiTrackLetList";
    const string SAMPLE_KEY3 = "MxpiFrameInfo";
}

APP_ERROR MxpiPassengerFlowEstimation ::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "MxpiPassengerFlowEstimation::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    statiscalResult = 0;
    // Get the property values by key
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get();
    std::shared_ptr<string> motNamePropSptr = std::static_pointer_cast<string>(configParamMap["motSource"]);
    motName = *motNamePropSptr.get();
    std::shared_ptr<string> descriptionMessageProSptr = std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
    descriptionMessage_ = *descriptionMessageProSptr.get();
    std::shared_ptr<string> x0 = std::static_pointer_cast<string>(configParamMap["x0"]);
    x0_ = *x0.get();
    std::shared_ptr<string> y0 = std::static_pointer_cast<string>(configParamMap["y0"]);
    y0_ = *y0.get();
    std::shared_ptr<string> x1 = std::static_pointer_cast<string>(configParamMap["x1"]);
    x1_ = *x1.get();
    std::shared_ptr<string> y1 = std::static_pointer_cast<string>(configParamMap["y1"]);
    y1_ = *y1.get();
    return APP_ERR_OK;
}

APP_ERROR MxpiPassengerFlowEstimation ::DeInit()
{
    LogInfo << "MxpiPassengerFlowEstimation::DeInit end.";
    return APP_ERR_OK;
}

APP_ERROR MxpiPassengerFlowEstimation ::SetMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,
    const MxpiErrorInfo mxpiErrorInfo)
{
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

APP_ERROR MxpiPassengerFlowEstimation::PrintMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,
    MxpiErrorInfo mxpiErrorInfo, APP_ERROR app_error, std::string errorName)
{
    ErrorInfo_ << GetError(app_error, pluginName_) << errorName;
    LogError << errorName;
    mxpiErrorInfo.ret = app_error;
    mxpiErrorInfo.errorInfo = ErrorInfo_.str();
    SetMxpiErrorInfo(buffer, pluginName_, mxpiErrorInfo);
    return app_error;
}


/*
 * @description: Replace className with trackId
 */
APP_ERROR MxpiPassengerFlowEstimation::GenerateOutput(const MxpiObjectList srcMxpiObjectList,
                                                      const MxpiTrackLetList srcMxpiTrackLetList,
                                                      const MxpiFrameInfo srcMxpiFrameInfo,
                                                      MxpiObjectList& dstMxpiObjectList)
{
    int x0 = atoi(x0_.c_str());
    int y0 = atoi(y0_.c_str());
    int x1 = atoi(x1_.c_str());
    int y1 = atoi(y1_.c_str());
    int passengerNumThisFrame = 0;
    for (int i = 0; i < srcMxpiObjectList.objectvec_size(); i++) {
        MxpiObject srcMxpiObject = srcMxpiObjectList.objectvec(i);
        MxpiClass srcMxpiClass = srcMxpiObject.classvec(0);
        MxpiObject* dstMxpiObject = dstMxpiObjectList.add_objectvec();
        dstMxpiObject->set_x0(srcMxpiObject.x0());
        dstMxpiObject->set_y0(srcMxpiObject.y0());
        dstMxpiObject->set_x1(srcMxpiObject.x1());
        dstMxpiObject->set_y1(srcMxpiObject.y1());
        MxpiClass* dstMxpiClass = dstMxpiObject->add_classvec();
        dstMxpiClass->set_confidence(srcMxpiClass.confidence());
        for (int j = 0; j < srcMxpiTrackLetList.trackletvec_size(); j++) {
            MxpiTrackLet srcMxpiTrackLet = srcMxpiTrackLetList.trackletvec(j);
            int INTEGER = 2;
            if (srcMxpiTrackLet.trackflag() != INTEGER) {
                MxpiMetaHeader srcMxpiHeader = srcMxpiTrackLet.headervec(0);
                if (srcMxpiHeader.memberid() == i) {
                    dstMxpiClass->set_classid(0);
                    dstMxpiClass->set_classname(to_string(srcMxpiTrackLet.trackid()));
                    continue;
                }
            }
        }
    }
    if (lastObjects.empty()) {
        APP_ERROR ret = UpdateLastObjectList(dstMxpiObjectList);
    }
    else {
        for (int i = 0; i < dstMxpiObjectList.objectvec_size(); i++) {
            MxpiObject dstMxpiObject = dstMxpiObjectList.objectvec(i);
            MxpiClass dstMxpiClass = dstMxpiObject.classvec(0);
            int x = (dstMxpiObject.x0() + dstMxpiObject.x1())/2;
            int y = (dstMxpiObject.y0() + dstMxpiObject.y1())/2;
            std::pair<int, int> point = std::make_pair(x, y);
            int TrackId = atoi(dstMxpiClass.classname().c_str());
            if (lastObjects.count(TrackId) > 0) {
                std::pair<int, int> LastPoint = lastObjects[TrackId];
                bool Intersect = IsIntersect(LastPoint.first, LastPoint.second, point.first, point.second, x0, y0, x1, y1);
                if (Intersect) {
                    statiscalResult++ ;
                }
            }  
        }
        APP_ERROR ret = UpdateLastObjectList(dstMxpiObjectList);
    }
    return APP_ERR_OK;
}

APP_ERROR MxpiPassengerFlowEstimation::UpdateLastObjectList(const MxpiObjectList dstMxpiObjectList) {
    lastObjects.clear();
    for (int i = 0; i < dstMxpiObjectList.objectvec_size(); i++) {
        MxpiObject dstMxpiObject = dstMxpiObjectList.objectvec(i);
        MxpiClass dstMxpiClass = dstMxpiObject.classvec(0);
        int x = (dstMxpiObject.x0() + dstMxpiObject.x1())/2;
        int y = (dstMxpiObject.y0() + dstMxpiObject.y1())/2;
        std::pair<int, int> point = std::make_pair(x, y);
        int TrackId = atoi(dstMxpiClass.classname().c_str());
        lastObjects[TrackId] = point;
    }
    return APP_ERR_OK;
}

APP_ERROR MxpiPassengerFlowEstimation::Process(std::vector<MxpiBuffer*>& mxpiBuffer) {
    LogInfo << "MxpiPassengerFlowEstimation::Process start";
    MxpiBuffer* buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, APP_ERR_COMM_FAILURE, "MxpiPassengerFlowEstimation process is not implemented");
    }
    shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(parentName_);  // Get the data from buffer
    shared_ptr<void> metadata2 = mxpiMetadataManager.GetMetadata(motName);
    shared_ptr<void> metadata3 = mxpiMetadataManager.GetMetadata("ReservedFrameInfo");
    if (metadata == nullptr) {
        shared_ptr<MxpiObjectList> dstMxpiObjectListSptr = make_shared<MxpiObjectList>();
        MxpiObject* dstMxpiObject = dstMxpiObjectListSptr->add_objectvec();
        MxpiClass* dstMxpiClass = dstMxpiObject->add_classvec();
        APP_ERROR ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiObjectListSptr));
        if (ret != APP_ERR_OK) {
            return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, ret, "MxpiPassengerFlowEstimation add metadata failed.");
        }
        SendData(0, *buffer); // Send the data to downstream plugin
        LogInfo << "MxpiPassengerFlowEstimation::Process end";
        return APP_ERR_OK;
    }
    google::protobuf::Message* msg = (google::protobuf::Message*)metadata.get();
    const google::protobuf::Descriptor* desc = msg->GetDescriptor();
    google::protobuf::Message* msg2 = (google::protobuf::Message*)metadata2.get();
    const google::protobuf::Descriptor* desc2 = msg2->GetDescriptor();
    google::protobuf::Message* msg3 = (google::protobuf::Message*)metadata3.get();
    const google::protobuf::Descriptor* desc3 = msg3->GetDescriptor();
    if (desc->name() != SAMPLE_KEY) {    // check whether the proto struct name is MxpiObjectList
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, APP_ERR_PROTOBUF_NAME_MISMATCH, "Proto struct name is not MxpiObjectList, failed");
    }
    if (desc2->name() != SAMPLE_KEY2) {   // check whether the proto struct name is MxpiTrackList
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, APP_ERR_PROTOBUF_NAME_MISMATCH, "Proto struct name is not MxpiTrackLetList, failed");
    }
    if (desc3->name() != SAMPLE_KEY3) {
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, APP_ERR_PROTOBUF_NAME_MISMATCH, "Proto struct name is not MxpiFrameInfo, failed");
    }
    shared_ptr<MxpiObjectList> srcMxpiObjectListSptr = static_pointer_cast<MxpiObjectList>(metadata);
    shared_ptr<MxpiTrackLetList> srcMxpiTrackLetListSptr = static_pointer_cast<MxpiTrackLetList>(metadata2);
    shared_ptr<MxpiFrameInfo> srcMxpiFrameInfoSptr = static_pointer_cast<MxpiFrameInfo>(metadata3);
    shared_ptr<MxpiObjectList> dstMxpiObjectListSptr = make_shared<MxpiObjectList>();
    APP_ERROR ret = GenerateOutput(*srcMxpiObjectListSptr, *srcMxpiTrackLetListSptr, *srcMxpiFrameInfoSptr, *dstMxpiObjectListSptr); // Generate sample output
    if (ret != APP_ERR_OK) {
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, ret, "MxpiPassengerFlowEstimation gets inference information failed.");
    }
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiObjectListSptr)); // Add Generated data to metedata
    if (ret != APP_ERR_OK) {
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, ret, "MxpiPassengerFlowEstimation add metadata failed.");
    }
    SendData(0, *buffer);  // Send the data to downstream plugin
    LogInfo << "MxpiPassengerFlowEstimation::Process end" << endl << "Statiscal result is :" << statiscalResult;
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiPassengerFlowEstimation::DefineProperties()
{
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
   
    auto parentNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "dataSource", "inputName", "the name of postprocessor", "mxpi_selectobject0", "NULL", "NULL" });

    auto motNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "motSource", "parentName", "the name of previous plugin", "mxpi_motsimplesortV20", "NULL", "NULL" });
    
    auto descriptionMessageProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "descriptionMessage", "message", "Description mesasge of plugin",  "This is PassengerFlowEstimation", "NULL", "NULL" });
    
    auto x0_Sptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "x0", "x0", "x0", "736", "NULL", "NULL" });
    
    auto y0_Sptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "y0", "y0", "y0", "191", "NULL", "NULL" });
    
    auto x1_Sptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "x1", "x0", "x0", "1870", "NULL", "NULL" });
    
    auto y1_Sptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "y1", "y0", "y0", "191", "NULL", "NULL" });

    properties.push_back(parentNameProSptr);
    properties.push_back(motNameProSptr);
    properties.push_back(descriptionMessageProSptr);
    properties.push_back(x0_Sptr);
    properties.push_back(y0_Sptr);
    properties.push_back(x1_Sptr);
    properties.push_back(y1_Sptr);

    return properties;
}

bool MxpiPassengerFlowEstimation::IsIntersect(int px1, int py1, int px2, int py2, int px3, int py3, int px4, int py4)
{
	bool flag = false;
	double d = (px2 - px1) * (py4 - py3) - (py2 - py1) * (px4 - px3);
	if (d != 0)
	{
		double r = ((py1 - py3) * (px4 - px3) - (px1 - px3) * (py4 - py3)) / d;
		double s = ((py1 - py3) * (px2 - px1) - (px1 - px3) * (py2 - py1)) / d;
		if ((r >= 0) && (r <= 1) && (s >= 0) && (s <= 1))
		{
			flag = true;
		}
	}
	return flag;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiPassengerFlowEstimation)

