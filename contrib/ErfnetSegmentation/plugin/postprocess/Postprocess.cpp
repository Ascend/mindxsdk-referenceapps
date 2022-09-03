/*
 * Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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
#include "cmath"
#include "opencv2/opencv.hpp"
#include "opencv2/core/mat.hpp"
#include "Postprocess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
using namespace MxBase;
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
using namespace cv;
namespace {
    const int WIDTH = 1024;
    const int HEIGHT = 512;
    const int CITYSACPESCLASS = 20;
    const cv::Vec3b color_map[] = {
      cv::Vec3b(128, 64, 128),
      cv::Vec3b(244, 35, 232),
      cv::Vec3b(70, 70, 70),
      cv::Vec3b(102, 102, 156),
      cv::Vec3b(190, 153, 153),
      cv::Vec3b(153, 153, 153),
      cv::Vec3b(250, 170, 30),
      cv::Vec3b(220, 220, 0),
      cv::Vec3b(107, 142, 35),
      cv::Vec3b(152, 251, 152),
      cv::Vec3b(70, 130, 180),
      cv::Vec3b(220, 20, 60),
      cv::Vec3b(255, 0, 0),
      cv::Vec3b(0, 0, 142),
      cv::Vec3b(0, 0, 70),
      cv::Vec3b(0, 60, 100),
      cv::Vec3b(0, 80, 100),
      cv::Vec3b(0, 0, 230),
      cv::Vec3b(119, 11, 32),
      cv::Vec3b(0, 0, 0),
  };
    const std::string INFER_RESULT_PATH = "./infer_result/";


    const string TENSOR_KEY = "MxpiTensorPackageList";
    const string VISION_KEY = "MxpiVisionList";
    const uint32_t YUV_BYTES_NU = 3;
    const uint32_t YUV_BYTES_DE = 2;
    const uint32_t OUTPUT_IMAGE_WIDTH = 480;
    const uint32_t OUTPUT_IMAGE_HEIGHT = 240;
    const float THRESHOLD_VALUE = 0.5; // threshold value
    const float FUSION_COEFFICIENT = 0.6; // the coefficient of picture fusion
    const int PIXEL_FORMAT = 12;  // MxbasePixelFormat type
    struct ImageInfo {
        int modelWidth;
        int modelHeight;
        int imgWidth;
        int imgHeight;
    };
}


// Decode MxpiTensorPackageList
void GetTensors(const MxTools::MxpiTensorPackageList tensorPackageList, std::vector<MxBase::TensorBase> &tensors)
{
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

//  PostProcess
APP_ERROR MxpiPostProcess::PostProcess(std::vector<MxBase::TensorBase> &inputTensors,
    uint32_t imgHeight, uint32_t imgWidth, cv::Mat &mask)
{
    LogInfo << "MxpiPostProcess::PostProcess start.";
    MxBase::TensorBase &tensor = inputTensors[0];
    std::vector<uint32_t> shape = tensor.GetShape();

    // model is N*H*W*C
    uint32_t outputModelChannel = tensor.GetShape()[3];  // there is two channel, the first channel is not road, the second is road
    uint32_t outputModelWidth = tensor.GetShape()[2];   // get width value
    uint32_t outputModelHeight = tensor.GetShape()[1];  // get height value
    cv::Mat imageMat(outputModelHeight, outputModelWidth, CV_32FC1);
    auto data = reinterpret_cast<float (*)[outputModelWidth][outputModelChannel]>(tensor.GetBuffer());

    for (size_t x = 0; x < outputModelHeight; ++x) {
        for (size_t y = 0; y < outputModelWidth; ++y) {
            imageMat.at<float>(x, y) = data[x][y][1];  // The probability of identifying it as road
        }
    }

    cv::resize(imageMat, imageMat, cv::Size(imgWidth, imgHeight), cv::INTER_CUBIC);
    cv::Mat argmax(imgHeight, imgWidth, CV_8UC1);
    const int WHITE = 255, BLACK = 0;
    for (size_t x = 0; x < imgHeight; ++x) {
        for (size_t y = 0; y < imgWidth; ++y) {
            // if probability more than threshold value is true
            argmax.at<uchar>(x, y) = (imageMat.at<float>(x, y) > THRESHOLD_VALUE) ? WHITE: BLACK;
        }
    }
    mask = argmax;
    LogInfo << "MxpiPostProcess::PostProcess end.";
    return APP_ERR_OK;
}

// get original image and fuse with mask ,output visioList
APP_ERROR MxpiPostProcess::GenerateVisionList(const cv::Mat mask, MxpiVisionList& dstMxpiVisionList)
{
    MxTools::MxpiVision dstVision;
    APP_ERROR ret = Mat2MxpiVision(0, mask, dstVision);
    if (ret != APP_ERR_OK) {
        LogWarn << "element("<< elementName_<<")  ImageFusion failed";
    }
    dstMxpiVisionList.add_visionvec()->CopyFrom(dstVision);
    if (dstMxpiVisionList.visionvec_size() == 0) {
        LogError <<  "element("<< elementName_<<") dst vision vec size is 0!";
        return APP_ERR_COMM_FAILURE;
    }
    return APP_ERR_OK;
}
APP_ERROR MxpiPostProcess::Mat2MxpiVision(size_t idx, const cv::Mat& mat, MxTools::MxpiVision& vision)
{
    auto header = vision.add_headervec();
    header->set_memberid(idx);
    header->set_datasource(parentName_);
    auto visionInfo = vision.mutable_visioninfo();
    visionInfo->set_format(PIXEL_FORMAT);
    visionInfo->set_height(mat.rows);
    visionInfo->set_heightaligned(mat.rows);
    visionInfo->set_width(mat.cols);
    visionInfo->set_widthaligned(mat.cols);
    auto visionData = vision.mutable_visiondata();
    visionData->set_datasize(mat.cols * mat.rows * mat.elemSize());
    MemoryData memoryDataDst(visionData->datasize(), MemoryData::MEMORY_HOST_MALLOC, deviceId_);
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


APP_ERROR MxpiPostProcess::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "MxpiPostProcess::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    // Get the property values by key 插件对应的属性值将通过“configParamMap”入参传入，可通过属性名称获取。
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get();
    this->index = 0;
    return APP_ERR_OK;
}

APP_ERROR MxpiPostProcess::DeInit()
{
    LogInfo << "MxpiPostProcess::DeInit end.";
    return APP_ERR_OK;
}

APP_ERROR MxpiPostProcess::SetMxpiErrorInfo(MxpiBuffer& buffer,
    const std::string pluginName, const MxpiErrorInfo mxpiErrorInfo)
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

APP_ERROR MxpiPostProcess::GenerateVisionListOutput(const MxpiTensorPackageList srcMxpiTensorPackageList,
    MxpiVisionList& dstMxpiVisionList)
{
    // Get Tensor
    std::vector<MxBase::TensorBase> tensors = {};
    GetTensors(srcMxpiTensorPackageList, tensors);
    MxBase::TensorBase *tensor = &tensors[0];
    cv::Mat imgrgb = cv::Mat(HEIGHT, WIDTH, CV_8UC3);
    // 1 x 20 x 512 x 1024
    auto data = reinterpret_cast<float *>(tensor->GetBuffer());
    float inferPixel[CITYSACPESCLASS];
    for (size_t x = 0; x < HEIGHT; ++x) {
        for (size_t y = 0; y < WIDTH; ++y) {
        for (size_t c = 0; c < CITYSACPESCLASS; ++c) {
            inferPixel[c] = *(data + c * WIDTH * HEIGHT + x * WIDTH + y);  // c, x, y
        }
        size_t max_index = std::max_element(inferPixel, inferPixel + CITYSACPESCLASS) - inferPixel;
        imgrgb.at<cv::Vec3b>(x, y) = color_map[max_index];
        }
    }
    LogInfo << INFER_RESULT_PATH + std::to_string(this->index) + ".png" << " saved !";
    cv::imwrite(INFER_RESULT_PATH + std::to_string(this->index++) + ".png", imgrgb);
}

APP_ERROR MxpiPostProcess::Process(std::vector<MxpiBuffer*>& mxpiBuffer)
{
    LogInfo << "MxpiPostProcess::Process start";
    MxpiBuffer* buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);     // Get metadata by key
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) << "MxpiSamplePlugin process is not implemented";
        mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        LogError << "MxpiPostProcess process is not implemented";
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

    // check whether the proto struct name is MxpiTensorPackageList
    google::protobuf::Message* msg = (google::protobuf::Message*)metadata.get();
    const google::protobuf::Descriptor* desc = msg->GetDescriptor();
    if (desc->name() != TENSOR_KEY) {
        ErrorInfo_ << GetError(APP_ERR_PROTOBUF_NAME_MISMATCH, pluginName_)
                   << "Proto struct name is not MxpiTensorPackageList, failed";
        mxpiErrorInfo.ret = APP_ERR_PROTOBUF_NAME_MISMATCH;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_PROTOBUF_NAME_MISMATCH; // self define the error code
    }
    
    // Generate VisionList output
    shared_ptr<MxpiTensorPackageList> srcMxpiTensorPackageListSptr = static_pointer_cast<MxpiTensorPackageList>(metadata);
    shared_ptr<MxpiVisionList> dstMxpiVisionListSptr = make_shared<MxpiVisionList>();
    APP_ERROR ret = GenerateVisionListOutput(*srcMxpiTensorPackageListSptr, *dstMxpiVisionListSptr);
    LogInfo << "MxpiPostProcess::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiPostProcess::DefineProperties()
{
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto parentNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
            STRING, "dataSource", "name", "the name of previous plugin", "mxpi_tensorlinfer0", "NULL", "NULL"});
    properties.push_back(parentNameProSptr);
    return properties;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiPostProcess)

