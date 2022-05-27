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
#include "MxpiCollisionClassName.h"
#include "MxBase/Log/Log.h"
#include <math.h>
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
namespace {
    const string SAMPLE_KEY = "MxpiObjectList";
}

APP_ERROR MxpiCollisionClassName::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "MxpiCollisionClassName::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    // Get the property values by key
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get();
    std::shared_ptr<string> descriptionMessageProSptr = std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
    descriptionMessage_ = *descriptionMessageProSptr.get();
    return APP_ERR_OK;
}

APP_ERROR MxpiCollisionClassName::DeInit()
{
    LogInfo << "MxpiCollisionClassName::DeInit end.";
    return APP_ERR_OK;
}

APP_ERROR MxpiCollisionClassName::SetMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,const MxpiErrorInfo mxpiErrorInfo)
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

APP_ERROR MxpiCollisionClassName::GenerateSampleOutput(const MxpiObjectList srcMxpiObjectList, MxpiObjectList& dstMxpiObjectList)   //Åö×²ÅÐ¶¨
{
    int n=srcMxpiObjectList.objectvec_size();
    double data[n][3];
    double num[n][n];
    for (int i = 0; i < n; i++)
    {
        data[i][0]=(srcMxpiObjectList.objectvec(i).x0()+srcMxpiObjectList.objectvec(i).x1())/2;
        data[i][1]=(srcMxpiObjectList.objectvec(i).y0()+srcMxpiObjectList.objectvec(i).y1())/2;
        data[i][2]=pow(pow(srcMxpiObjectList.objectvec(i).x1()-srcMxpiObjectList.objectvec(i).x0(),2)+pow(srcMxpiObjectList.objectvec(i).y1()-srcMxpiObjectList.objectvec(i).y0(),2),0.5)/2;
    }
    
    int a[n*n][2];  //´æ´¢Åö×²±àºÅ
    for (int i = 0; i < n*n; i++)
    {a[i][0]=-1;a[i][1]=-1;}
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            num[i][j]=pow(pow(data[i][0]-data[j][0],2)+pow(data[i][1]-data[j][1],2),0.5);
            if (num[i][j] < max(data[i][2],data[j][2]))
            {
                if (i < j)
                {
                    a[(i+1)*(j+1)-1][0]=i;a[(i+1)*(j+1)-1][1]=j;
                }
            }
        }
    }        
    
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
        //dstMxpiClass->set_classid(0);
        int t=0;
        for (int o = 0; o < n*n; o++)
        {
            for (int p = 0; p < 2; p++)
            {
                if (a[o][p]==i)
                    t++;
            }
        }
        if (t==0)
        {
            dstMxpiClass->set_classname(srcMxpiClass.classname());    //¸´ÖÆ
        }
        else
        {
            dstMxpiClass->set_classname("Collision");
        }
   
    }
    
    return APP_ERR_OK;
}

APP_ERROR MxpiCollisionClassName::Process(std::vector<MxpiBuffer*>& mxpiBuffer)
{
    LogInfo << "MxpiCollisionClassName::Process start";
    MxpiBuffer* buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) << "MxpiCollisionClassName process is not implemented";
        mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        LogError << "MxpiCollisionClassName process is not implemented";
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
        ErrorInfo_ << GetError(APP_ERR_PROTOBUF_NAME_MISMATCH, pluginName_) << "Proto struct name is not MxpiObjectList, failed";
        mxpiErrorInfo.ret = APP_ERR_PROTOBUF_NAME_MISMATCH;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_PROTOBUF_NAME_MISMATCH; // self define the error code
    }
    // Generate sample output
    shared_ptr<MxpiObjectList> srcMxpiObjectListSptr = static_pointer_cast<MxpiObjectList>(metadata);
    shared_ptr<MxpiObjectList> dstMxpiObjectListSptr = make_shared<MxpiObjectList>();
    APP_ERROR ret = GenerateSampleOutput(*srcMxpiObjectListSptr, *dstMxpiObjectListSptr);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret, pluginName_) << "MxpiCollisionClassName gets inference information failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Add Generated data to metedata
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiObjectListSptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiCollisionClassName add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    LogInfo << "MxpiCollisionClassName::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiCollisionClassName::DefineProperties()
{
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto parentNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
        STRING, "dataSource", "name", "the name of previous plugin", "mxpi_trackidreplaceclassname0", "NULL", "NULL"});
    auto descriptionMessageProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
        STRING, "descriptionMessage", "message", "Description mesasge of plugin", "This is MxpiCollisionClassName", "NULL", "NULL"});
    properties.push_back(parentNameProSptr);
    properties.push_back(descriptionMessageProSptr);
    return properties;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiCollisionClassName)