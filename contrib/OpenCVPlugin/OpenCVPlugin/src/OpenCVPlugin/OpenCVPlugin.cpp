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

#include "OpenCVPlugin.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
using namespace MxBase;
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
namespace {
    const string SAMPLE_KEY = "MxpiVisionList";
    const int YUV_U = 2;
    const int YUV_V = 3;
    const int san = 3;
    const int er = 2;
    const int yi = 1;
}

APP_ERROR MxpiSamplePlugin::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "MxpiSamplePlugin::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    // Get the property values by key
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get();
    std::shared_ptr<string> descriptionMessageProSptr =
    std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
    height = *std::static_pointer_cast<float>(configParamMap["height"]);
    width = *std::static_pointer_cast<float>(configParamMap["width"]);
    fx = *std::static_pointer_cast<double>(configParamMap["fx"]);
    fy = *std::static_pointer_cast<double>(configParamMap["fy"]);
    interpolation = *std::static_pointer_cast<int>(configParamMap["interpolation"]); 
    startRow = *std::static_pointer_cast<double>(configParamMap["startRow"]);
    endRow = *std::static_pointer_cast<double>(configParamMap["Height"]);
    startCol = *std::static_pointer_cast<double>(configParamMap["startCol"]);
    endCol = *std::static_pointer_cast<double>(configParamMap["Width"]);
    outputDataFormat = *std::static_pointer_cast<string>(configParamMap["outputDataFormat"]);
    dataType = *std::static_pointer_cast<string>(configParamMap["dataType"]);
    option = *std::static_pointer_cast<string>(configParamMap["option"]);
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

APP_ERROR MxpiSamplePlugin::openCV(size_t idx, const MxTools::MxpiVision srcMxpiVision,
                                   MxTools::MxpiVision& dstMxpiVision)
{
    // init
    LogInfo << "opencv begin";
    auto& visionInfo = srcMxpiVision.visioninfo();
    auto& visionData = srcMxpiVision.visiondata();
    MxBase::MemoryData memorySrc = {};
    memorySrc.deviceId = visionData.deviceid();
    memorySrc.type = (MxBase::MemoryData::MemoryType) visionData.memtype();
    memorySrc.size = visionData.datasize();
    memorySrc.ptrData = (void*)visionData.dataptr();
    MxBase::MemoryData memoryDst(visionData.datasize(), MxBase::MemoryData::MEMORY_HOST_NEW);
    APP_ERROR  res = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst, memorySrc);
    if (res != APP_ERR_OK) {
        LogError << "Fail to malloc and copy host memory.";
        return res;
    }
    cv::Mat src;
    cv::Mat imgBgr = cv::Mat(visionInfo.heightaligned(), visionInfo.widthaligned(), CV_8UC3);
    if (memorySrc.type == san) {
	src = cv::Mat(srcMxpiVision.visioninfo().heightaligned(), srcMxpiVision.visioninfo().widthaligned(), CV_8UC3,
               memoryDst.ptrData);
    }
    else {
	src = cv::Mat(srcMxpiVision.visioninfo().heightaligned()* YUV_V / YUV_U, srcMxpiVision.visioninfo().widthaligned(), CV_8UC1,
               memoryDst.ptrData);
	cv::cvtColor(src, imgBgr, cv::COLOR_YUV2BGR_NV12);
    }
    cv::Mat dst;
    cv::Mat imgYuv;
    cv::Mat imgRgb;
    cv::Mat yuv_mat;
    cv::Mat img_nv12;
    MxBase::MemoryData memoryNewDst(dst.data, MxBase::MemoryData::MEMORY_HOST_NEW);
    outputPixelFormat_ = (MxBase::MxbasePixelFormat)srcMxpiVision.visioninfo().format();
    if (option == "resize") {
	if (memorySrc.type == er) {
	    cv::resize(imgBgr, dst, cv::Size(width, height), fx, fy, interpolation);
	}
	else {
	    cv::resize(src, dst, cv::Size(width, height), fx, fy, interpolation);
	}
    }
    else {
      cv::Rect ori(startRow, startCol, endRow, endCol);
      if (memorySrc.type == san) {
	dst = src(ori).clone();
	}
      else {
	dst = imgBgr(ori).clone();
      }
    }
    auto ret = APP_ERR_OK;
    if (outputDataFormat == "YUV") {
	height = dst.rows;
	width = dst.cols;
	imgYuv = cv::Mat(height, width, CV_8UC1);
    Bgr2Yuv(dst,imgYuv);
	auto ret = Mat2MxpiVisionDvpp(idx, imgYuv, dstMxpiVision);
    }
    else {
	if (outputDataFormat == "RGB") {
        if (dataType == 'float32') {
            imgRgb = cv::Mat(height, width, CV_32FC3);
        }
        else {
            imgRgb = cv::Mat(height, width, CV_8UC3);
        }
    cv::cvtColor(dst, imgRgb, cv::COLOR_BGR2RGB);
	auto ret = Mat2MxpiVisionOpencv(idx, imgRgb, dstMxpiVision);
	}
	else if (outputDataFormat == "BGR") {
	auto ret = Mat2MxpiVisionOpencv(idx, dst, dstMxpiVision);
	}
	else {
	LogError << "outputDataFormat not in RGB,BGR,YUV";
	}
    }
    if (ret != APP_ERR_OK) {
        LogError << "convert mat to mxvision failed!";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR MxpiSamplePlugin::Bgr2Yuv(cv::Mat src, cv::Mat &dst)
{
    int w_img = src.cols;
    int h_img = src.rows;
    dst = cv::Mat(h_img*1.5, w_img, CV_8UC1);
    cv::Mat src_YUV_I420(h_img*1.5, w_img, CV_8UC1);  //YUV_I420
    cvtColor(src, src_YUV_I420, cv::COLOR_BGR2YUV_I420);
    swapYUV_I420toNV12(src_YUV_I420.data, dst.data, w_img, h_img);
    return APP_ERR_OK;
}

void  MxpiSamplePlugin::swapYUV_I420toNV12(unsigned char* i420bytes, unsigned char* nv12bytes, int width, int height)
{
    int nLenY = width * height;
    int nLenU = nLenY / 4;

    memcpy(nv12bytes, i420bytes, width * height);

    for (int i = 0; i < nLenU; i++)
    {
        nv12bytes[nLenY + 2 * i] = i420bytes[nLenY + i];                    // U
        nv12bytes[nLenY + 2 * i + 1] = i420bytes[nLenY + nLenU + i];        // V
    }
}

APP_ERROR MxpiSamplePlugin::Mat2MxpiVisionDvpp(size_t idx, const cv::Mat& mat, MxTools::MxpiVision& vision)
{
    LogInfo << "Mat2MxpiVision begin";
    auto header = vision.add_headervec();
    header->set_memberid(idx);
    header->set_datasource(parentName_);

    auto visionInfo = vision.mutable_visioninfo();
    visionInfo->set_format(outputPixelFormat_);
    visionInfo->set_height(mat.rows*YUV_U/YUV_V);
    visionInfo->set_heightaligned(mat.rows*YUV_U/YUV_V);
    visionInfo->set_width(mat.cols);
    visionInfo->set_widthaligned(mat.cols);

    auto visionData = vision.mutable_visiondata();
    visionData->set_datasize(mat.cols * mat.rows * mat.elemSize());
    MemoryData memoryDataDst(visionData->datasize(), MemoryData::MEMORY_DVPP, deviceId_);
    MemoryData memoryDataStr(mat.data, visionData->datasize(), MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR  ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataStr);
    if (ret != APP_ERR_OK) {
        LogError << "copy memory error.";
        return ret;
    }
    visionData->set_dataptr((uint64)memoryDataDst.ptrData);
    visionData->set_deviceid(deviceId_);
    visionData->set_memtype(MxTools::MXPI_MEMORY_DVPP);
    if (dataType == "float32") {
        visionData->set_datatype(MxTools::MxpiDataType::MXPI_DATA_TYPE_FLOAT32);
    }
    else if (dataType == "uint8") {
        visionData->set_datatype(MxTools::MxpiDataType::MXPI_DATA_TYPE_UINT8);
    }
    else {
	LogError << "dataType must in float32,uint8";
    }
    LogInfo << "Mat2MxpiVision done";
    return APP_ERR_OK;
};

APP_ERROR MxpiSamplePlugin::Mat2MxpiVisionOpencv(size_t idx, const cv::Mat& mat, MxTools::MxpiVision& vision)
{
    LogInfo << "Mat2MxpiVision begin";
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
    MemoryData memoryDataDst(visionData->datasize(), MemoryData::MEMORY_HOST, deviceId_);
    MemoryData memoryDataStr(mat.data, visionData->datasize(), MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR  ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataStr);
    if (ret != APP_ERR_OK) {
        LogError << "copy memory error.";
        return ret;
    }
    visionData->set_dataptr((uint64)memoryDataDst.ptrData);
    visionData->set_deviceid(deviceId_);
    visionData->set_memtype(MxTools::MXPI_MEMORY_HOST);
    if (dataType == "float32") {
	visionData->set_datatype(MxTools::MxpiDataType::MXPI_DATA_TYPE_FLOAT32);
    }
    else if (dataType == "uint8") {
	visionData->set_datatype(MxTools::MxpiDataType::MXPI_DATA_TYPE_UINT8);
    }
    else {
        LogError << "dataType must in float32,uint8";
    }
    LogInfo << "Mat2MxpiVision done";
    return APP_ERR_OK;
};

APP_ERROR MxpiSamplePlugin::GenerateVisionList(const MxpiVisionList srcMxpiVisionList,
                                               MxpiVisionList& dstMxpiVisionList)
{
    for (int i = 0; i< srcMxpiVisionList.visionvec_size();i++) {
        auto srcMxpiVision = srcMxpiVisionList.visionvec(i);
        MxTools::MxpiVision dstVision;
        APP_ERROR ret = openCV(i, srcMxpiVision, dstVision);
        if (ret != APP_ERR_OK) {
            LogWarn << "element("<< elementName_<<") normailze failed";
        }
        dstMxpiVisionList.add_visionvec()->CopyFrom(dstVision);
    }
    if (dstMxpiVisionList.visionvec_size() == 0) {
        LogError <<  "element("<< elementName_<<") dst vision vec size is 0!";
        return APP_ERR_COMM_FAILURE;
    }
    LogInfo << "Generate done";
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
    LogInfo << "generate";
    APP_ERROR ret = GenerateVisionList(*srcMxpiVisionListSptr, *dstMxpiVisionListptr);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret, pluginName_) << "MxpiSamplePlugin gets inference information failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Add Generated data to metedata
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiVisionListptr));
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
    auto parentNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "dataSource", "name", "the name of previous plugin", "mxpi_imageresize0", "NULL", "NULL"});
    auto descriptionMessageProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "descriptionMessage", "message", "Description mesasge of plugin", "This is MxpiSamplePlugin", "NULL", "NULL"});
    auto startRow = std::make_shared<ElementProperty<double>>(ElementProperty<double> {
	DOUBLE, "startRow", "startRow", "the start_row of crop image", 1, 0.0, 8192.0});
    auto startCol = std::make_shared<ElementProperty<double>>(ElementProperty<double> {
	DOUBLE, "startCol", "startCol", "the start_col of crop  image", 1, 0.0, 8192.0});
    auto Height = std::make_shared<ElementProperty<double>>(ElementProperty<double> {
	DOUBLE, "Height", "Height", "the Height of crop image", 256, 1.0, 8192.0});
    auto Width = std::make_shared<ElementProperty<double>>(ElementProperty<double> {
	DOUBLE, "Width", "Width", "the Width of crop image", 256, 1.0, 8192.0});
    auto height = std::make_shared<ElementProperty<float>>(ElementProperty<float> {
        FLOAT, "height", "height", "the height of image", 256, 1, 8192});
    auto width = std::make_shared<ElementProperty<float>>(ElementProperty<float> {
        FLOAT, "width", "width", "the width of image", 256, 1, 8192});
    auto fx = std::make_shared<ElementProperty<double>>(ElementProperty<double> {
        DOUBLE, "fx", "fx", "the fx ratio  of image", 0, 0, 1});
    auto fy = std::make_shared<ElementProperty<double>>(ElementProperty<double> {
        DOUBLE, "fy", "fy", "the fy ratio  of image", 0, 0, 1});
    auto outputDataFormat = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "outputDataFormat", "outputDataFormat", "the format of the output  RGB or BGR or YUV", "YUV", "NULL", "NULL"});
    auto dataType = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "dataType", "dataType", "the dataType  float32 or uint8 ", "uint8", "NULL", "NULL"});
    auto interpolation = std::make_shared<ElementProperty<int>>(ElementProperty<int> {
        INT, "interpolation", "interpolation", "the interpolation  of image", 1, 0, 4});
    auto option = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
		            STRING, "option", "option", "OPTION of plugin", "resize", "NULL", "NULL"});
    properties.push_back(parentNameProSptr);
    properties.push_back(descriptionMessageProSptr);
    properties.push_back(startRow);
    properties.push_back(startCol);
    properties.push_back(Height);
    properties.push_back(Width);
    properties.push_back(height);
    properties.push_back(width);
    properties.push_back(fx);
    properties.push_back(fy);
    properties.push_back(outputDataFormat);
    properties.push_back(dataType);
    properties.push_back(option);
    properties.push_back(interpolation);

    return properties;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiSamplePlugin)
