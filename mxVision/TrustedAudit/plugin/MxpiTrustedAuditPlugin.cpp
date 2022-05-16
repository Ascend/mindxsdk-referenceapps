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
#include "MxpiTrustedAuditPlugin.h"
#include "MxBase/Log/Log.h"
#include "cstdlib"
#include <unistd.h>
#include <string>
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
namespace {
    const string SAMPLE_KEY = "MxpiObjectList";
}
#define PATH_MAX 255
#define SLEEP_TIME 10

APP_ERROR MxpiTrustedAuditPlugin::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "MxpiTrustedAuditPlugin::Init start.";
    string version = "1.0";
    char current_path[PATH_MAX];
    if (!getcwd(current_path, PATH_MAX)) {
        LogInfo << "get current path fail!!";
    }
    const char *ptr;
    ptr = strrchr(current_path, '/');
    int length = strlen(current_path) - strlen(ptr);
    string s_current_path(current_path);
    string s_last_pth;
    s_last_pth = s_current_path.substr(0, length);
    LogInfo << s_last_pth;

    string instruction = "docker restart container_elasticsearch_" + version;
    system((const char*)instruction.c_str());
    instruction = "docker restart container_gauss_" + version;
    system((const char*)instruction.c_str());
    instruction = "docker restart container_python_" + version;
    system((const char*)instruction.c_str());
    sleep(SLEEP_TIME);

    instruction = "docker exec container_python_" + version + " python -u /home/tranlog_audit_serv.py >> /tmp/server.log 2>&1 &";
    LogInfo << instruction;
    system((const char*)instruction.c_str());
    sleep(SLEEP_TIME);

    std::shared_ptr<string> originalLogsPathProSptr = std::static_pointer_cast<string>(configParamMap["originalLogsPath"]);
    originalLogsPath_ = *originalLogsPathProSptr.get();

    instruction = "python -u " + s_last_pth + "/mindx/mindx_watcher_and_sender.py >> /tmp/watcher.log 2>&1 &";
    LogInfo << instruction;
    system((const char*)instruction.c_str());

    LogInfo << "MxpiTrustedAuditPlugin::Init end.";
    APP_ERROR ret = APP_ERR_OK;
    // Get the property values by key
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get();
    std::shared_ptr<string> descriptionMessageProSptr = std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
    descriptionMessage_ = *descriptionMessageProSptr.get();
    return APP_ERR_OK;
}

APP_ERROR MxpiTrustedAuditPlugin::DeInit()
{
    LogInfo << "MxpiTrustedAuditPlugin::DeInit start.";
    string version = "1.0";
    char current_path[PATH_MAX];
    if (!getcwd(current_path, PATH_MAX)) {
        LogInfo << "get current path fail!!";
    }
    const char *ptr;
    ptr = strrchr(current_path, '/');
    int length = strlen(current_path) - strlen(ptr);
    string s_current_path(current_path);
    string s_last_pth;
    s_last_pth = s_current_path.substr(0, length);
    LogInfo << s_last_pth;

    string instruction = "python -u " + s_last_pth + "/mindx/kill_watcher.py >> /tmp/kill_watcher.log 2>&1 &";
    LogInfo << instruction;
    system((const char*)instruction.c_str());

    LogInfo << originalLogsPath_;
    instruction = "rm " + originalLogsPath_ + "/*";
    system((const char*)instruction.c_str());

    instruction = "docker stop container_python_" + version;
    system((const char*)instruction.c_str());
    instruction = "docker stop container_elasticsearch_" + version;
    system((const char*)instruction.c_str());
    instruction = "docker stop container_gauss_" + version;
    system((const char*)instruction.c_str());

    LogInfo << "MxpiTrustedAuditPlugin::DeInit end.";
    return APP_ERR_OK;
}

APP_ERROR MxpiTrustedAuditPlugin::SetMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName, const MxpiErrorInfo mxpiErrorInfo)
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

APP_ERROR MxpiTrustedAuditPlugin::GenerateSampleOutput(const MxpiObjectList srcMxpiObjectList, MxpiClassList& dstMxpiClassList)
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

APP_ERROR MxpiTrustedAuditPlugin::Process(std::vector<MxpiBuffer*>& mxpiBuffer)
{
    LogInfo << "MxpiTrustedAuditPlugin::Process start";
    MxpiBuffer* buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) << "MxpiTrustedAuditPlugin process is not implemented";
        mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        LogError << "MxpiTrustedAuditPlugin process is not implemented";
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
    shared_ptr<MxpiClassList> dstMxpiClassListSptr = make_shared<MxpiClassList>();
    APP_ERROR ret = GenerateSampleOutput(*srcMxpiObjectListSptr, *dstMxpiClassListSptr);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret, pluginName_) << "MxpiTrustedAuditPlugin gets inference information failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Add Generated data to metedata
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiClassListSptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiTrustedAuditPlugin add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    LogInfo << "MxpiTrustedAuditPlugin::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiTrustedAuditPlugin::DefineProperties()
{
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto parentNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "dataSource", "name", "the name of previous plugin", "mxpi_modelinfer0", "NULL", "NULL"
    });
    auto descriptionMessageProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "descriptionMessage", "message", "Description mesasge of plugin", "This is MxpiTrustedAuditPlugin", "NULL", "NULL"
    });
    auto originalLogsPathProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "originalLogsPath", "originalLogsPath", "the path of the original MindX logs", "/work/mindx_sdk/mxVision/logs", "NULL", "NULL"
    });

    properties.push_back(parentNameProSptr);
    properties.push_back(descriptionMessageProSptr);
    properties.push_back(originalLogsPathProSptr);

    return properties;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiTrustedAuditPlugin)
