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

#include "MxpiTrackIdReplaceClassName.h"
#include "MxBase/Log/Log.h"
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
namespace {
    const string SAMPLE_KEY = "MxpiObjectList";
    const string SAMPLE_KEY2 = "MxpiTrackLetList";
}

APP_ERROR MxpiTrackIdReplaceClassName::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "MxpiTrackIdReplaceClassName::Init start.";
    APP_ERROR ret = APP_ERR_OK;

    // Get the property values by key
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get();
    std::shared_ptr<string> motNamePropSptr = std::static_pointer_cast<string>(configParamMap["motSource"]);
    motName_ = *motNamePropSptr.get();   
    std::shared_ptr<string> descriptionMessageProSptr = std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
    descriptionMessage_ = *descriptionMessageProSptr.get();
    return APP_ERR_OK;
}

APP_ERROR MxpiTrackIdReplaceClassName::DeInit()
{
    LogInfo << "MxpiTrackIdReplaceClassName::DeInit end.";
    return APP_ERR_OK;
}

APP_ERROR MxpiTrackIdReplaceClassName::SetMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,
    const MxpiErrorInfo mxpiErrorInfo)
{
    APP_ERROR ret = APP_ERR_OK;
    // Define an object of MxpiMetadataManager
    MxpiMetadataManager mxpiMetadataManager(buffer);
    ret = mxpiMetadataManager.AddErrorInfo(pluginName, mxpiErrorInfo);
    if (ret != APP_ERR_OK) {
        // LogError << "Failed to AddErrorInfo.";
        std::cout << "Failed to AddErrorInfo." << std::endl;
        // return ret;
    }
    ret = SendData(0, buffer);
    return ret;
}

APP_ERROR MxpiTrackIdReplaceClassName::GenerateSampleOutput(const MxpiObjectList srcMxpiObjectList, const MxpiTrackLetList srcMxpiTrackLetList,
    MxpiObjectList& dstMxpiObjectList)
{
    for (int i = 0; i < srcMxpiObjectList.objectvec_size(); i++){
        MxpiObject srcMxpiObject = srcMxpiObjectList.objectvec(i);   //得到输入srcMxpiObjectList的objectvec
        MxpiClass srcMxpiClass = srcMxpiObject.classvec(0);        //得到输入srcMxpiObjectList的objectvec的classvec
        MxpiObject* dstMxpiObject = dstMxpiObjectList.add_objectvec();    //给输出添加objectvec
        dstMxpiObject->set_x0(srcMxpiObject.x0());
        dstMxpiObject->set_y0(srcMxpiObject.y0());
        dstMxpiObject->set_x1(srcMxpiObject.x1());
        dstMxpiObject->set_y1(srcMxpiObject.y1());
        MxpiClass* dstMxpiClass = dstMxpiObject->add_classvec();   //给输出添加objectvec
        dstMxpiClass->set_confidence(srcMxpiClass.confidence());
        for(int j = 0; j < srcMxpiTrackLetList.trackletvec_size(); j++){
            MxpiTrackLet srcMxpiTrackLet = srcMxpiTrackLetList.trackletvec(j);  //得到输入srcMxpiTrackLetList的trackletvec
            if(srcMxpiTrackLet.trackflag() != 2){
                MxpiMetaHeader srcMxpiHeader = srcMxpiTrackLet.headervec(0);  //得到输入srcMxpiTrackLetList的trackletvec的headervec
                if(srcMxpiHeader.memberid() == i){
                    // dstMxpiClass->set_classid(srcMxpiTrackLet.trackid());
                    dstMxpiClass->set_classid(0);
                    dstMxpiClass->set_classname(to_string(srcMxpiTrackLet.trackid()));
                    continue;
                }
            }
        }
    }
}


APP_ERROR MxpiTrackIdReplaceClassName::Process(std::vector<MxpiBuffer*>& mxpiBuffer)
{
    LogInfo << "MxpiTrackIdReplaceClassName::Process start";
    MxpiBuffer* buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) << "MxpiTrackIdReplaceClassName process is not implemented";
        mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        LogError << "MxpiTrackIdReplaceClassName process is not implemented";
        return APP_ERR_COMM_FAILURE;
    }

    // Get the data from buffer
    shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(parentName_);
    shared_ptr<void> metadata2 = mxpiMetadataManager.GetMetadata(motName_);
    if (metadata == nullptr) {
        ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "Metadata is NULL, failed";
        mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_METADATA_IS_NULL; // self define the error code
    }    
    if (metadata2 == nullptr) {
        ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "Metadata is NULL, failed";
        mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        // if(SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo) == 0 ){
        //     std::cout<<"Failed to AddErrorInfo."<<std::endl;
        // }
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
        return APP_ERR_PROTOBUF_NAME_MISMATCH; // self define the error code
    }
    // check whether the proto struct name is MxpiTrackList
    google::protobuf::Message* msg2 = (google::protobuf::Message*)metadata2.get();
    const google::protobuf::Descriptor* desc2 = msg2->GetDescriptor();
    if (desc2->name() != SAMPLE_KEY2) {        
        ErrorInfo_ << GetError(APP_ERR_PROTOBUF_NAME_MISMATCH, pluginName_) 
            << "Proto struct name is not MxpiTrackLetList, failed";
        mxpiErrorInfo.ret = APP_ERR_PROTOBUF_NAME_MISMATCH;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        std::cout<<" SetMxpiErrorInfo: "<<SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo)<<std::endl;
        return APP_ERR_PROTOBUF_NAME_MISMATCH; // self define the error code
    }



    // Generate sample output
    shared_ptr<MxpiObjectList> srcMxpiObjectListSptr = static_pointer_cast<MxpiObjectList>(metadata);
    shared_ptr<MxpiTrackLetList> srcMxpiTrackLetListSptr = static_pointer_cast<MxpiTrackLetList>(metadata2);
    shared_ptr<MxpiObjectList> dstMxpiObjectListSptr = make_shared<MxpiObjectList>();    
    APP_ERROR ret = GenerateSampleOutput(*srcMxpiObjectListSptr,*srcMxpiTrackLetListSptr,*dstMxpiObjectListSptr);

    if (ret != APP_ERR_OK) {
        LogError << GetError(ret, pluginName_) << "MxpiTrackIdReplaceClassName gets inference information failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }

    // Add Generated data to metedata
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiObjectListSptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiTrackIdReplaceClassName add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    LogInfo << "MxpiTrackIdReplaceClassName::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiTrackIdReplaceClassName::DefineProperties()
{
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
   
    auto parentNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
        STRING, "dataSource", "inputName", "the name of objectpostprocessor", "mxpi_objectpostprocessor0", "NULL", "NULL"});
    auto motNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
        STRING, "motSource", "parentName", "the name of previous plugin", "mxpi_motsimplesortV20", "NULL", "NULL"});

    auto descriptionMessageProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
        STRING, "descriptionMessage", "message", "Description mesasge of plugin", "This is MxpiTrackIdReplaceClassName", "NULL", "NULL"});
    properties.push_back(parentNameProSptr);
    properties.push_back(motNameProSptr);
    
    properties.push_back(descriptionMessageProSptr);
    return properties;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiTrackIdReplaceClassName)

