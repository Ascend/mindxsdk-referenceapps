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

#include "cmath"
#include "MxpiPFLDPostProcessPlugin.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
using namespace MxBase;
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
namespace {
    const string SAMPLE_KEY = "MxpiTensorPackageList";
}

// Decode MxpiTensorPackageList
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

APP_ERROR MxpiPFLDPostProcessPlugin::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "MxpiPFLDPostProcessPlugin::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    // Get the property values by key
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get();
    return APP_ERR_OK;
}

APP_ERROR MxpiPFLDPostProcessPlugin::DeInit()
{
    LogInfo << "MxpiPFLDPostProcessPlugin::DeInit end.";
    return APP_ERR_OK;
}

APP_ERROR MxpiPFLDPostProcessPlugin::SetMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,
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

APP_ERROR MxpiPFLDPostProcessPlugin::GenerateObjectList(const MxpiTensorPackageList srcMxpiTensorPackage,MxpiObjectList& dstMxpiObjectList)
{
    // Get Tensor
    std::vector<MxBase::TensorBase> tensors = {};
    GetTensors(srcMxpiTensorPackage, tensors);
    // Get the coordinates of eyes and mouth
    auto dataPtr = (uint8_t *)tensors[0].GetBuffer();
    float* eyes_left = new float[20];
    float* eyes_right = new float[20];
    float* mouth = new float[40];
    for(int i = 0; i < 20; i++)
    {
        eyes_left[i] = *((float*)dataPtr + i + 66);
    }
    for(int i = 0; i < 20; i++)
    {
        eyes_right[i] = *((float*)dataPtr + i + 174);
    }
    for(int i = 0; i < 40; i++)
    {
        mouth[i] = *((float*)dataPtr + i + 104);
    }
    // Calculate the MAR(Mouth Aspect Ratio) of person
    float MAR = (sqrt(pow(fabs(mouth[5] - mouth[29]), 2) + pow(fabs(mouth[4] - mouth[28]), 2)) + sqrt(pow(fabs(mouth[11] - mouth[37]), 2) + pow(fabs(mouth[10] - mouth[36]), 2))) / (2 * sqrt(pow(fabs(mouth[19] - mouth[1]), 2) + pow(fabs(mouth[18] - mouth[0]), 2)));
    // Generate an ObjectList to save relevant information
    MxpiObject* dstMxpiObject = dstMxpiObjectList.add_objectvec();
    MxpiMetaHeader* dstMxpiMetaHeaderList = dstMxpiObject->add_headervec();
    dstMxpiMetaHeaderList->set_datasource(parentName_);
    dstMxpiMetaHeaderList->set_memberid(0);
    /**
     * @x0 MAR
     * @y0 Height of the left eye
     * @x1 Width of the right eye
     * @y1 Height of the right eye
     */
    dstMxpiObject->set_x0(MAR);
    dstMxpiObject->set_y0(sqrt(pow(fabs(eyes_left[1] - eyes_left[15]), 2) + pow(fabs(eyes_left[0] - eyes_left[14]), 2)));
    dstMxpiObject->set_x1(fabs(eyes_right[12] - eyes_right[4]));
    dstMxpiObject->set_y1(sqrt(pow(fabs(eyes_right[1] - eyes_right[15]), 2) + pow(fabs(eyes_right[0] - eyes_right[14]), 2)));
    // Release dynamic array
    delete []eyes_left;
    delete []eyes_right;
    delete []mouth;
    return APP_ERR_OK;
}

APP_ERROR MxpiPFLDPostProcessPlugin::Process(std::vector<MxpiBuffer*>& mxpiBuffer)
{
    LogInfo << "MxpiPFLDPostProcessPlugin::Process start";
    MxpiBuffer* buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) << "MxpiPFLDPostProcessPlugin process is not implemented";
        mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        LogError << "MxpiPFLDPostProcessPlugin process is not implemented";
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
    // check the proto struct name
    google::protobuf::Message* msg = (google::protobuf::Message*)metadata.get();
    const google::protobuf::Descriptor* desc = msg->GetDescriptor();
    if (desc->name() != SAMPLE_KEY) {
        ErrorInfo_ << GetError(APP_ERR_PROTOBUF_NAME_MISMATCH, pluginName_) 
        << "Proto struct name is not MxpiTensorPackageList, failed";
        mxpiErrorInfo.ret = APP_ERR_PROTOBUF_NAME_MISMATCH;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_PROTOBUF_NAME_MISMATCH; // self define the error code
    }
    // Generate output
    shared_ptr<MxpiTensorPackageList> srcMxpiTensorPackageListSptr = static_pointer_cast<MxpiTensorPackageList>(metadata);
    shared_ptr<MxpiObjectList> dstMxpiObjectListSptr = make_shared<MxpiObjectList>();
    APP_ERROR ret = GenerateObjectList(*srcMxpiTensorPackageListSptr, *dstMxpiObjectListSptr);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret, pluginName_) << "MxpiPostProcessPlugin gets inference information failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Add Generated data to metedata
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiObjectListSptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiPFLDPostProcessPlugin add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    LogInfo << "MxpiPFLDPostProcessPlugin::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiPFLDPostProcessPlugin::DefineProperties()
{
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto parentNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
        STRING, "dataSource", "name", "the name of previous plugin", "mxpi_tensorinfer1", "NULL", "NULL"});
    properties.push_back(parentNameProSptr);
    return properties;
}

// Register the PFLDPostProcess plugin through macro
MX_PLUGIN_GENERATE(MxpiPFLDPostProcessPlugin)
