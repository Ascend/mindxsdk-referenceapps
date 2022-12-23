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

#include "MxpiSamplePlugin.h"
#include "MxBase/Log/Log.h"
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
namespace {
    const string SAMPLE_KEY = "MxpiObjectList";
    const int SAMPLE_INT = 648;
    const string SAMPLE_STRING = "sample proton string with id 648";
}

APP_ERROR MxpiSamplePlugin::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "MxpiSamplePlugin::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    // Get the property values by key
    parentName_ = dataSource_;
    std::shared_ptr<string> descriptionMessageProSptr = 
        std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
    descriptionMessage_ = *descriptionMessageProSptr.get();
    return APP_ERR_OK;
}

APP_ERROR MxpiSamplePlugin::DeInit()
{
    LogInfo << "MxpiSamplePlugin::DeInit end.";
    return APP_ERR_OK;
}

APP_ERROR MxpiSamplePlugin::SetMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,
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

APP_ERROR MxpiSamplePlugin::GenerateSampleOutput(const MxpiObjectList srcMxpiObjectList, 
    MxpiClassList& dstMxpiClassList)
{
    for (int i = 0; i < srcMxpiObjectList.objectvec_size(); i++) {
        MxpiObject srcMxpiObject = srcMxpiObjectList.objectvec(i);
        for (int j = 0; j < srcMxpiObject.classvec_size(); j++) {
            MxpiClass srcMxpiClass = srcMxpiObject.classvec(j);
            MxpiClass* dstMxpiClass = dstMxpiClassList.add_classvec();
            MxpiMetaHeader* dstMxpiMetaHeaderList = dstMxpiClass->add_headervec();
            dstMxpiMetaHeaderList->set_datasource(parentName_);
            dstMxpiMetaHeaderList->set_memberid(j);
            dstMxpiClass->set_classid(srcMxpiClass.classid());
            dstMxpiClass->set_confidence(srcMxpiClass.confidence());
            dstMxpiClass->set_classname(srcMxpiClass.classname() + "," + descriptionMessage_);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR MxpiSamplePlugin::Process(std::vector<MxpiBuffer*>& mxpiBuffer)
{
    LogInfo << "MxpiSamplePlugin::Process start";
    MxpiBuffer* buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) << "MxpiSamplePlugin process is not implemented";
        mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        LogError << "MxpiSamplePlugin process is not implemented";
        return APP_ERR_COMM_FAILURE;
    }
    // Get the data from buffer
    shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(parentName_);
    if (metadata == nullptr) {
        ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "Metadata is NULL, failed";
        mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_METADATA_IS_NULL; // self define the error code
    }
    // check whether the proto struct name is MxpiObjectList
    google::protobuf::Message* msg = (google::protobuf::Message*)metadata.get();
    const google::protobuf::Descriptor* desc = msg->GetDescriptor();
    if (desc->name() != SAMPLE_KEY) {
        ErrorInfo_ << GetError(APP_ERR_PROTOBUF_NAME_MISMATCH, pluginName_) 
            << "Proto struct name is not MxpiObjectList, failed";
        mxpiErrorInfo.ret = APP_ERR_PROTOBUF_NAME_MISMATCH;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_PROTOBUF_NAME_MISMATCH; // self define the error code
    }
    // Generate sample output
    shared_ptr<MxpiObjectList> srcMxpiObjectListSptr = static_pointer_cast<MxpiObjectList>(metadata);
    shared_ptr<MxpiClassList> dstMxpiClassListSptr = make_shared<MxpiClassList>();
    APP_ERROR ret = GenerateSampleOutput(*srcMxpiObjectListSptr, *dstMxpiClassListSptr);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret, pluginName_) << "MxpiSamplePlugin gets inference information failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }

    // Generate sample proto
    auto mxpiSampleProtoListptr = std::make_shared<sampleproto::MxpiSampleProtoList>();
    auto mxpiSampleProtoptr = mxpiSampleProtoListptr->add_sampleprotovec();

    sampleproto::MxpiMetaHeader* dstMxpiMetaHeaderList = mxpiSampleProtoptr->add_headervec();
    dstMxpiMetaHeaderList->set_datasource(parentName_);
    dstMxpiMetaHeaderList->set_memberid(0);

    mxpiSampleProtoptr->set_intsample(SAMPLE_INT);
    mxpiSampleProtoptr->set_stringsample(SAMPLE_STRING);

    std::string sampleProtoName = "mxpi_sampleproto";
    ret = mxpiMetadataManager.AddProtoMetadata(sampleProtoName, static_pointer_cast<void>(mxpiSampleProtoListptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, sampleProtoName) << "MxpiSamplePlugin add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, sampleProtoName, mxpiErrorInfo);
        return ret;
    }

    // Add Generated data to metedata
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiClassListSptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiSamplePlugin add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    LogInfo << "MxpiSamplePlugin::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiSamplePlugin::DefineProperties()
{
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto descriptionMessageProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
        STRING, "descriptionMessage", "message", "Description mesasge of plugin", "This is MxpiSamplePlugin", "NULL", "NULL"});
    properties.push_back(descriptionMessageProSptr);
    return properties;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiSamplePlugin)
