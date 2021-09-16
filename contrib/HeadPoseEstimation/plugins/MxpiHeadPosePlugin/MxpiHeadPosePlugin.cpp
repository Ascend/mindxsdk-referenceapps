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

#include "MxpiHeadPosePlugin.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"

using namespace MxBase;
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
namespace {
    const string SAMPLE_KEY = "MxpiTensorPackageList";
    const int SAMPLE_CLASS_ID = 42;
    const float SAMPLE_CONF = 0.314;
    const string SAMPLE_CLASS_NAME = "The shape of tensor[0] in metadata is ";
}

// decode MxpiTensorPackageList
void GetTensors(const MxTools::MxpiTensorPackageList tensorPackageList,
                std::vector<MxBase::TensorBase> &tensors) {
    for (int i = 0; i < tensorPackageList.tensorpackagevec_size(); ++i) {
        for (int j = 0; j < tensorPackageList.tensorpackagevec(i).tensorvec_size(); j++) {
            MxBase::MemoryData memoryData = {};
            memoryData.deviceId = tensorPackageList.tensorpackagevec(i).tensorvec(j).deviceid();
            memoryData.type = (MxBase::MemoryData::MemoryType)tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).memtype();
            memoryData.size = (uint32_t) tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).tensordatasize();
            memoryData.ptrData = (void *) tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).tensordataptr();
            std::vector<uint32_t> outputShape = {};
            for (int k = 0; k < tensorPackageList.
            tensorpackagevec(i).tensorvec(j).tensorshape_size(); ++k) {
                outputShape.push_back((uint32_t) tensorPackageList.
                tensorpackagevec(i).tensorvec(j).tensorshape(k));
            }
            MxBase::TensorBase tmpTensor(memoryData, true, outputShape,
                                         (MxBase::TensorDataType)tensorPackageList.
                                         tensorpackagevec(i).tensorvec(j).tensordatatype());
            tensors.push_back(tmpTensor);
        }
    }
}

/**
 * @brief Parsing TensorBase data to keypoint heatmap and PAF heatmap of openpose model
 * @param tensors - TensorBase vector
 * @param keypoint_heatmap - Keep keypoint data
 * @param paf_heatmap - Keep PAF data
 * @param channel_keypoint - Channel number of keypoint heatmap
 * @param channel_paf - Channel number of PAF heatmap
 * @param height - Height of two heatmaps
 * @param width - Width of two heatmaps
 */
// void ReadDataFromTensor(const std::vector <MxBase::TensorBase> &tensors,
//                                             std::vector<std::vector<std::vector<float> > > &keypoint_heatmap,
//                                             std::vector<std::vector<std::vector<float> > > &paf_heatmap,
//                                             int channel_keypoint, int channel_paf, int height, int width) {
//     auto dataPtr = (uint8_t *)tensors[1].GetBuffer();
//     std::shared_ptr<void> keypoint_pointer;
//     keypoint_pointer.reset(dataPtr, uint8Deleter);
//     int idx = 0;
//     float temp_data = 0.0;
//     for (int i = 0; i < channel_keypoint; i++) {
//         for (int j = 0; j < height; j++) {
//             for (int k = 0; k < width;  k++) {
//                 temp_data = static_cast<float *>(keypoint_pointer.get())[idx];
//                 if (temp_data < 0) {
//                     temp_data = 0;
//                 }
//                 keypoint_heatmap[i][j][k] = temp_data;
//                 idx += 1;
//             }
//         }
//     }

//     auto data_paf_ptr = (uint8_t *)tensors[0].GetBuffer();
//     std::shared_ptr<void> paf_pointer;
//     paf_pointer.reset(data_paf_ptr, uint8Deleter);
//     idx = 0;
//     temp_data = 0.0;
//     for (int i = 0; i < channel_paf; i++) {
//         for (int j = 0; j < height; j++) {
//             for (int k = 0; k < width;  k++) {
//                 temp_data = static_cast<float *>(paf_pointer.get())[idx];
//                 paf_heatmap[i][j][k] = temp_data;
//                 idx += 1;
//             }
//         }
//     }
// }

APP_ERROR MxpiHeadPosePlugin::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "MxpiHeadPosePlugin::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    // Get the property values by key
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get();
    LogInfo << "MxpiHeadPosePlugin::Init complete.";
    return APP_ERR_OK;
}

APP_ERROR MxpiHeadPosePlugin::DeInit()
{
    LogInfo << "MxpiHeadPosePlugin::DeInit start.";
    LogInfo << "MxpiHeadPosePlugin::DeInit complete.";
    return APP_ERR_OK;
}

APP_ERROR MxpiHeadPosePlugin::SetMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,
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

APP_ERROR MxpiHeadPosePlugin::GenerateHeadPoseInfo(const MxpiTensorPackageList srcMxpiTensorPackage,
                                                    mxpiheadposeproto::MxpiHeadPoseList& dstMxpiHeadPoseList)
{
    // Get Tensor
    std::vector<MxBase::TensorBase> tensors = {};
    GetTensors(srcMxpiTensorPackage, tensors);
    if (tensors.size() == 1) {
        // tensorflow model
        auto headpose = tensors[0].GetShape();
        LogWarn << "source Tensor number:" << tensors.size() << endl;
        LogWarn << "Tensor[0] ByteSize in .cpp:" << tensors[0].GetByteSize() << endl;
        LogWarn << "headpose[0]:" << headpose[0] << endl;
        // LogWarn << "headpose[1]:" << headpose[1] << endl;
        // LogWarn << "headpose[2]:" << headpose[2] << endl;
    }
    else {
        LogWarn << "Tensor Size Error!!" << endl;
        LogWarn << "source Tensor number:" << tensors.size() << endl;
        LogWarn << "Tensor[0] ByteSize in .cpp:" << tensors[0].GetByteSize() << endl;
    }

    // Generate HeadPoseInfo
    
    auto dstMxpiHeadPoseInfoPtr = dstMxpiHeadPoseList.add_headposeinfovec();
    mxpiheadposeproto::MxpiMetaHeader* dstMxpiMetaHeaderList = dstMxpiHeadPoseInfoPtr->add_headervec();
    dstMxpiMetaHeaderList->set_datasource(parentName_);
    dstMxpiMetaHeaderList->set_memberid(0);
    dstMxpiHeadPoseInfoPtr->set_yaw(SAMPLE_CLASS_ID);
    // dstMxpiClass->set_confidence(SAMPLE_CONF);
    // std::string OutputClassStr =  SAMPLE_CLASS_NAME + std::to_string(tensors[0].GetByteSize());
    // dstMxpiClass->set_classname(OutputClassStr + ", " + descriptionMessage_);

    return APP_ERR_OK;
}

APP_ERROR MxpiHeadPosePlugin::Process(std::vector<MxpiBuffer*>& mxpiBuffer)
{
    LogInfo << "MxpiHeadPosePlugin::Process start";
    MxpiBuffer* buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) << "MxpiHeadPosePlugin process is not implemented";
        mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        LogError << "MxpiHeadPosePlugin process is not implemented";
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



    // Generate WHENet output
    shared_ptr<MxpiTensorPackageList> srcMxpiTensorPackageListSptr = static_pointer_cast<MxpiTensorPackageList>(metadata);
    shared_ptr<mxpiheadposeproto::MxpiHeadPoseList> dstMxpiHeadPoseListSptr = make_shared<mxpiheadposeproto::MxpiHeadPoseList>();
    APP_ERROR ret = GenerateHeadPoseInfo(*srcMxpiTensorPackageListSptr, *dstMxpiHeadPoseListSptr);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret, pluginName_) << "MxpiHeadPosePlugin gets inference information failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }



    // Add Generated data to metedata
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiHeadPoseListSptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiHeadPosePlugin add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    LogInfo << "MxpiHeadPosePlugin::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiHeadPosePlugin::DefineProperties()
{
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto parentNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
        STRING, "dataSource", "name", "the name of previous plugin", "mxpi_tensorinfer1", "NULL", "NULL"});
    properties.push_back(parentNameProSptr);
    return properties;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiHeadPosePlugin)
