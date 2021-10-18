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

void Softmax(std::vector<float> myVector, std::vector<float> &result){
    float maxPosition = *max_element(myVector.begin(), myVector.end());
    std::vector<double> a;
    double b = 0;
    for(float& element : myVector){
        element -= maxPosition;
        a.push_back(exp(element));
        b += exp(element);
    }
    for(double& element : a)
        result.push_back(element*1.0/b);
}

APP_ERROR MxpiHeadPosePlugin::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "MxpiHeadPosePlugin::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    // Get the property values by key
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get();
    std::shared_ptr<string> descriptionMessageProSptr = 
        std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
    descriptionMessage_ = *descriptionMessageProSptr.get();
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
    if (tensors.size() == 3) {

        // Get output shape of model
        auto headpose1 = tensors[0].GetShape();
        auto headpose2 = tensors[1].GetShape();
        auto headpose3 = tensors[2].GetShape();

        // Generate yaw,pitch,roll
        auto yaw_dataPtr = (float *)tensors[0].GetBuffer();
        std::vector<float> myyaw, yaw_predicted_vec;
        for(int i=0; i<headpose1[1]; i++){
            myyaw.push_back(yaw_dataPtr[i]);
        }
        Softmax(myyaw, yaw_predicted_vec);
        float yaw_predicted = 0;
        for(int i=0; i<headpose1[1]; i++){
            yaw_predicted += yaw_predicted_vec[i] * i;
        }
        yaw_predicted = yaw_predicted * 3 - 180;

        auto pitch_dataPtr = (float *)tensors[1].GetBuffer();
        std::vector<float> mypitch, pitch_predicted_vec;
        for(int i=0; i<headpose2[1]; i++){
            mypitch.push_back(pitch_dataPtr[i]);
        }
        Softmax(mypitch, pitch_predicted_vec);
        float pitch_predicted = 0;
        for(int i=0; i<headpose2[1]; i++){
            pitch_predicted += pitch_predicted_vec[i] * i;
        }
        pitch_predicted = pitch_predicted * 3 - 99;

        auto roll_dataPtr = (float *)tensors[2].GetBuffer();
        std::vector<float> myroll, roll_predicted_vec;
        for(int i=0; i<headpose2[1]; i++){
            myroll.push_back(roll_dataPtr[i]);
        }
        Softmax(myroll, roll_predicted_vec);
        float roll_predicted = 0;
        for(int i=0; i<headpose3[1]; i++){
            roll_predicted += roll_predicted_vec[i] * i;
        }
        roll_predicted = roll_predicted * 3 - 99;

        // Generate HeadPoseInfo
        auto dstMxpiHeadPoseInfoPtr = dstMxpiHeadPoseList.add_headposeinfovec();
        mxpiheadposeproto::MxpiMetaHeader* dstMxpiMetaHeaderList = dstMxpiHeadPoseInfoPtr->add_headervec();
        dstMxpiMetaHeaderList->set_datasource(parentName_);
        dstMxpiMetaHeaderList->set_memberid(0);
        dstMxpiHeadPoseInfoPtr->set_yaw(yaw_predicted);
        dstMxpiHeadPoseInfoPtr->set_pitch(pitch_predicted);
        dstMxpiHeadPoseInfoPtr->set_roll(roll_predicted);
    }
    else {
        LogWarn << "Tensor Size Error!!" << endl;
        LogWarn << "source Tensor number:" << tensors.size() << endl;
        LogWarn << "Tensor[0] ByteSize in .cpp:" << tensors[0].GetByteSize() << endl;
    }

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
    // Check whether the proto struct name is MxpiObjectList
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
    auto descriptionMessageProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
        STRING, "descriptionMessage", "message", "Description mesasge of plugin", "This is MxpiSamplePlugin", "NULL", "NULL"});
    properties.push_back(parentNameProSptr);
    properties.push_back(descriptionMessageProSptr);
    return properties;
}

// Register the Sample plugin through macro
    MX_PLUGIN_GENERATE(MxpiHeadPosePlugin)
