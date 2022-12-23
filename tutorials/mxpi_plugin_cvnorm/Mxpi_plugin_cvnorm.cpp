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

#include "Mxpi_plugin_cvnorm.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
using namespace MxBase;
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
namespace {
    const string SAMPLE_KEY = "MxpiVisionList";
}

APP_ERROR MxpiPluginCvnorm::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "MxpiPluginCvnorm::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    // Get the property values by key
    parentName_ = dataSource_;
    std::shared_ptr<string> descriptionMessageProSptr = 
        std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
    descriptionMessage_ = *descriptionMessageProSptr.get();
    return APP_ERR_OK;
}

APP_ERROR MxpiPluginCvnorm::DeInit()
{
    LogInfo << "MxpiPluginCvnorm::DeInit end.";
    return APP_ERR_OK;
}

APP_ERROR MxpiPluginCvnorm::SetMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,
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
APP_ERROR MxpiPluginCvnorm::openCVNormailze(size_t idx,const MxTools::MxpiVision srcMxpiVision,
                           MxTools::MxpiVision& dstMxpiVision)
{
    // main func in this block!
    // init !!! 此处输入为dvpp解码的图像或帧，格式为YUV。如使用opencv需要自行修改格式
    cv::Mat src(srcMxpiVision.visioninfo().height(),srcMxpiVision.visioninfo().width(),CV_8UC3,
                (void*)srcMxpiVision.visiondata().dataptr());
    cv::Mat dst;
    // 如后续模型包含aipp，则如例保持YUV格式
    outputPixelFormat_ = (MxBase::MxbasePixelFormat)srcMxpiVision.visioninfo().format(); 
    
    // normailze 
    cv::normalize(src, dst, 255, 0, cv::NORM_MINMAX, CV_8U);

    // mat2vision
    auto ret = Mat2MxpiVision(idx, dst ,dstMxpiVision);
    if (ret != APP_ERR_OK) {
        LogError << "convert mat to mxvision failed!";
        return ret;
    }
    return APP_ERR_OK;
};
APP_ERROR MxpiPluginCvnorm::Mat2MxpiVision(size_t idx, const cv::Mat& mat ,MxTools::MxpiVision& vision)
{
    auto header = vision.add_headervec();
    header->set_memberid(idx);
    header->set_datasource(parentName_);

    auto visionInfo = vision.mutable_visioninfo();
    visionInfo->set_format(outputPixelFormat_);
    visionInfo->set_height(mat.rows);
    visionInfo->set_heightaligned(mat.rows);
    visionInfo->set_width(mat.cols);
    visionInfo->set_widthaligned(mat.cols);

    auto visionData = vision.mutable_visiondata();
    visionData->set_datasize(mat.cols * mat.rows * mat.elemSize());
    MemoryData memoryDataDst(visionData->datasize(), MemoryData::MEMORY_HOST_MALLOC,deviceId_);
    MemoryData memoryDataStr(mat.data, visionData->datasize(), MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR  ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataStr);
    if (ret != APP_ERR_OK) {
        LogError << "copy memory error.";
        return ret;
    }
    visionData->set_dataptr((uint64)memoryDataDst.ptrData);
    visionData->set_deviceid(deviceId_);
    visionData->set_memtype(MxTools::MXPI_MEMORY_HOST_MALLOC);
    visionData->set_datatype(MxTools::MxpiDataType::MXPI_DATA_TYPE_UINT8);

    return APP_ERR_OK;
};
APP_ERROR MxpiPluginCvnorm::GenerateVisionList(const MxpiVisionList srcMxpiVisionList,
                                              MxpiVisionList& dstMxpiVisionList)
{
    LogInfo <<"input type:" <<srcMxpiVisionList.visionvec(0).visiondata().datatype();
    for (int i = 0; i< srcMxpiVisionList.visionvec_size();i++) {
        auto srcMxpiVision = srcMxpiVisionList.visionvec(i);
        MxTools::MxpiVision dstVision;
        APP_ERROR ret = openCVNormailze(i,srcMxpiVision,dstVision);
        if (ret != APP_ERR_OK) {
            LogWarn << "element("<< elementName_<<") normailze failed";
        }
        dstMxpiVisionList.add_visionvec()->CopyFrom(dstVision);
    }
    if (dstMxpiVisionList.visionvec_size() == 0) {
        LogError <<  "element("<< elementName_<<") dst vision vec size is 0!";
        return APP_ERR_COMM_FAILURE;
    }

    return APP_ERR_OK;
}

APP_ERROR MxpiPluginCvnorm::Process(std::vector<MxpiBuffer*>& mxpiBuffer)
{
    LogInfo << "MxpiPluginCvnorm::Process start";
    MxpiBuffer* buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) << "MxpiPluginCvnorm process is not implemented";
        mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        LogError << "MxpiPluginCvnorm process is not implemented";
        return APP_ERR_COMM_FAILURE;
    }
    // Get the data from buffer
    shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(parentName_);
    if (metadata == nullptr) {
        SendData(0, *buffer);
        return APP_ERR_METADATA_IS_NULL; // self define the error code
    }
    // check the proto struct name
    google::protobuf::Message* msg = (google::protobuf::Message*)metadata.get();
    const google::protobuf::Descriptor* desc = msg->GetDescriptor();
    if (desc->name() != SAMPLE_KEY) {
        ErrorInfo_ << GetError(APP_ERR_PROTOBUF_NAME_MISMATCH, pluginName_) 
        << "Proto struct name is not MxpiVisionList, failed with:" << desc->name();
        mxpiErrorInfo.ret = APP_ERR_PROTOBUF_NAME_MISMATCH;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_PROTOBUF_NAME_MISMATCH; // self define the error code
    }
    // Generate sample output
    shared_ptr<MxpiVisionList> srcMxpiVisionListSptr = static_pointer_cast<MxpiVisionList>(metadata);
    shared_ptr<MxpiVisionList> dstMxpiVisionListptr = make_shared<MxpiVisionList>();
    APP_ERROR ret = GenerateVisionList(*srcMxpiVisionListSptr, *dstMxpiVisionListptr);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret, pluginName_) << "MxpiPluginCvnorm gets inference information failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Add Generated data to metedata
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiVisionListptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiPluginCvnorm add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    LogInfo << "MxpiPluginCvnorm::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiPluginCvnorm::DefineProperties()
{
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto descriptionMessageProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
        STRING, "descriptionMessage", "message", "Description mesasge of plugin", "This is plugin for cvnorm func", "NULL", "NULL"});
    properties.push_back(descriptionMessageProSptr);
    return properties;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(Mxpi_plugin_cvnorm)
