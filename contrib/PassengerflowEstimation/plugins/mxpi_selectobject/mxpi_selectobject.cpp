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
#include <string.h>
#include "mxpi_selectobject.h"
#include "MxBase/Log/Log.h"
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
namespace {
    const string SAMPLE_KEY = "MxpiObjectList";
}

APP_ERROR MxpiSelectObject::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "MxpiSelectObject::Init start.";
    APP_ERROR ret = APP_ERR_OK;

    // Get the property values by key
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get(); 
    std::shared_ptr<string> descriptionMessageProSptr = 
        std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
    descriptionMessage_ = *descriptionMessageProSptr.get();
    return APP_ERR_OK;
}

APP_ERROR MxpiSelectObject::DeInit()
{
    LogInfo << "MxpiSelectObject::DeInit end.";
    return APP_ERR_OK;
}

APP_ERROR MxpiSelectObject::SetMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,
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

APP_ERROR MxpiSelectObject::PrintMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,
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
APP_ERROR MxpiSelectObject::GenerateSampleOutput(const MxpiObjectList srcMxpiObjectList, 
                                                 MxpiObjectList& dstMxpiObjectList)
{
    for (int i = 0; i < srcMxpiObjectList.objectvec_size(); i++) {
        const char *target1 = "person";
        const char *target2 = "car";
        const int maxArea = 160000; 
        MxpiObject srcMxpiObject = srcMxpiObjectList.objectvec(i);       
        MxpiClass srcMxpiClass = srcMxpiObject.classvec(0);
        int Area = abs(srcMxpiObject.x0() - srcMxpiObject.x1()) * abs(srcMxpiObject.y0() - srcMxpiObject.y1());
        if ((!strcmp(srcMxpiClass.classname().c_str(),target1) || !strcmp(srcMxpiClass.classname().c_str(),target2)) 
        && Area < maxArea){
            MxpiObject* dstMxpiObject = dstMxpiObjectList.add_objectvec();    
            dstMxpiObject->set_x0(srcMxpiObject.x0());
            dstMxpiObject->set_y0(srcMxpiObject.y0());
            dstMxpiObject->set_x1(srcMxpiObject.x1());
            dstMxpiObject->set_y1(srcMxpiObject.y1());
            MxpiClass* dstMxpiClass = dstMxpiObject->add_classvec();   
            dstMxpiClass->set_confidence(srcMxpiClass.confidence());
            dstMxpiClass->set_classid(0);
            dstMxpiClass->set_classname(srcMxpiClass.classname());
        }
    }
    return APP_ERR_OK;
}

APP_ERROR MxpiSelectObject::Process(std::vector<MxpiBuffer*>& mxpiBuffer){
    LogInfo << "MxpiSelectObject::Process start";
    MxpiBuffer* buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, APP_ERR_COMM_FAILURE, "MxpiSelectObject process is not implemented");
    }
    shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(parentName_);
    if (metadata == nullptr) {
        shared_ptr<MxpiObjectList> dstMxpiObjectListSptr = make_shared<MxpiObjectList>(); 
        MxpiObject* dstMxpiObject = dstMxpiObjectListSptr->add_objectvec();   
        MxpiClass* dstMxpiClass = dstMxpiObject->add_classvec();    
        APP_ERROR ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiObjectListSptr));
        if (ret != APP_ERR_OK) {
            return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, ret, "MxpiSelectObject add metadata failed.");
        }
        SendData(0, *buffer); // Send the data to downstream plugin
        LogInfo << "MxpiSelectObject::Process end";
        return APP_ERR_OK;
    }
    google::protobuf::Message* msg = (google::protobuf::Message*)metadata.get();
    const google::protobuf::Descriptor* desc = msg->GetDescriptor();
    if (desc->name() != SAMPLE_KEY) {    // check whether the proto struct name is MxpiObjectList
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, APP_ERR_PROTOBUF_NAME_MISMATCH, "Proto struct name is not MxpiObjectList, failed");
    }
    shared_ptr<MxpiObjectList> srcMxpiObjectListSptr = static_pointer_cast<MxpiObjectList>(metadata);
    shared_ptr<MxpiObjectList> dstMxpiObjectListSptr = make_shared<MxpiObjectList>();   
    APP_ERROR ret = GenerateSampleOutput(*srcMxpiObjectListSptr,*dstMxpiObjectListSptr); // Generate sample output
    if (ret != APP_ERR_OK) {
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, ret, "MxpiSelectObject select failed.");
    }
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiObjectListSptr)); // Add Generated data to metedata
    if (ret != APP_ERR_OK) {
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, ret, "MxpiSelectObject add metadata failed.");
    }
    SendData(0, *buffer);  // Send the data to downstream plugin
    LogInfo << "MxpiSelectObject::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiSelectObject::DefineProperties()
{
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
   
    auto parentNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
        STRING, "dataSource", "inputName", "the name of objectpostprocessor", "mxpi_objectpostprocessor0", "NULL", "NULL"});

    auto descriptionMessageProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
        STRING, "descriptionMessage", "message", "Description mesasge of plugin",  "This is MxpiSelectObject", "NULL", "NULL"});

    properties.push_back(parentNameProSptr);
    properties.push_back(descriptionMessageProSptr);
    return properties;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiSelectObject)

