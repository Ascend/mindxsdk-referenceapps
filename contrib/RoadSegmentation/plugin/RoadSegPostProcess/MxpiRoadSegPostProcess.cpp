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
#include "cmath"
#include "opencv2/opencv.hpp"
#include "opencv2/core/mat.hpp"
#include "MxpiRoadSegPostProcess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
using namespace MxBase;
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
using namespace cv;
namespace {
    const string TENSOR_KEY = "MxpiTensorPackageList";
    const string VISION_KEY = "MxpiVisionList";
    const uint32_t YUV_BYTES_NU = 3;
    const uint32_t YUV_BYTES_DE = 2;
    const float THRESHOLD_VALUE = 0.5; // threshold value
    const float FUSION_COEFFICIENT = 0.7; // the coefficient of picture fusion
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
// convert chw_to_hwc
APP_ERROR MxpiRoadSegPostProcess::chw2hwc(const std::vector<MxBase::TensorBase> inputTensors,
                                          std::vector<MxBase::TensorBase> &outputTensors)
{
    LogInfo << "MxpiRoadSegPostProcess::chw_to_hwc start.";
    if (inputTensors.size() == 0) {
        LogInfo << "inputTensors is 0";
    }
    auto tensor = inputTensors[0];
    auto inputShape = tensor.GetShape();
    uint32_t N = inputShape[0], C = inputShape[1], H = inputShape[2], W = inputShape[3];
    std::vector<uint32_t> outputShape = {N, H, W, C};
    MxBase::TensorBase tmpTensor(outputShape, tensor.GetDataType());
    APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tmpTensor);
    if (ret != APP_ERR_OK) {
        LogInfo << "TensorBaseMalloc error";
        return ret;
    }
    for (uint32_t i = 0; i < N; i++) {
        auto tensorPtr = (float*)tensor.GetBuffer() + i * tensor.GetByteSize() / N;
        auto tmpTensorPtr = (float*)tmpTensor.GetBuffer() + i * tmpTensor.GetByteSize() / N;
        uint32_t stride = H * W;
        for (uint32_t c = 0; c != C; ++c) {
            uint32_t t = c * stride;
            for (uint32_t j = 0; j != stride; ++j) {
                float f = *(tensorPtr+t+j);
                *(tmpTensorPtr+j * (C)+c) = f;
            }
        }
    }
    outputTensors.push_back(tmpTensor);
    LogInfo << "MxpiRoadSegPostProcess::chw_to_hwc end.";
    return APP_ERR_OK;
}

//  PostProcess
APP_ERROR MxpiRoadSegPostProcess::PostProcess(std::vector<MxBase::TensorBase> &inputTensors,
                                              const ImageInfo &imageInfo, cv::Mat &mask)
{
    LogInfo << "MxpiRoadSegPostProcess::PostProcess start.";
    MxBase::TensorBase &tensor = inputTensors[0];
    std::vector<uint32_t> shape = tensor.GetShape();
    uint32_t imgHeight = imageInfo.imgHeight;
    uint32_t imgWidth = imageInfo.imgWidth;

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
    LogInfo << "MxpiRoadSegPostProcess::PostProcess end.";
    return APP_ERR_OK;
}

// get original image and fuse with mask ,output visioList
APP_ERROR MxpiRoadSegPostProcess::GenerateVisionList(const cv::Mat mask,
    const MxpiVisionList srcMxpiVisionList, MxpiVisionList& dstMxpiVisionList)
{
    LogInfo <<"input type:" <<srcMxpiVisionList.visionvec(0).visiondata().datatype();
    for (int i = 0; i< srcMxpiVisionList.visionvec_size();i++) {
        auto srcMxpiVision = srcMxpiVisionList.visionvec(i);
        MxTools::MxpiVision dstVision;
        APP_ERROR ret = openCVImageFusion(i, srcMxpiVision, dstVision, mask);
        if (ret != APP_ERR_OK) {
            LogWarn << "element("<< elementName_<<")  ImageFusion failed";
        }
        dstMxpiVisionList.add_visionvec()->CopyFrom(dstVision);
    }
    if (dstMxpiVisionList.visionvec_size() == 0) {
        LogError <<  "element("<< elementName_<<") dst vision vec size is 0!";
        return APP_ERR_COMM_FAILURE;
    }
    return APP_ERR_OK;
}
// image fusion
APP_ERROR MxpiRoadSegPostProcess::openCVImageFusion(size_t idx, const MxTools::MxpiVision srcMxpiVision,
                                                    MxTools::MxpiVision& dstMxpiVision,
                                                    cv::Mat threeChannelMask)
{
    LogInfo << " MxpiRoadSegPostProcess::openCVImageFusion start.";
    // init
    auto& visionInfo = srcMxpiVision.visioninfo();
    auto& visionData = srcMxpiVision.visiondata();
    MxBase::MemoryData memorySrc = {};
    memorySrc.deviceId = visionData.deviceid();
    memorySrc.type = (MxBase::MemoryData::MemoryType) visionData.memtype();
    memorySrc.size = visionData.datasize();
    memorySrc.ptrData = (void*)visionData.dataptr();
    MxBase::MemoryData memoryDst(visionData.datasize(), MxBase::MemoryData::MEMORY_HOST_NEW);
    APP_ERROR  ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst, memorySrc);
    if (ret != APP_ERR_OK) {
        LogError << "Fail to malloc and copy host memory.";
        return ret;
    }
    // Initialize the OpenCV image information matrix with the output original information
    cv::Mat imgYuv = cv::Mat(visionInfo.heightaligned()  * YUV_BYTES_NU / YUV_BYTES_DE,
                             visionInfo.widthaligned(), CV_8UC1, memoryDst.ptrData);
    cv::Mat imgBgr = cv::Mat(visionInfo.heightaligned(), visionInfo.widthaligned(), CV_8UC3);

    // YUV420sp to BGR
    cv::cvtColor(imgYuv, imgBgr, cv::COLOR_YUV2BGR_NV21);
    
    // crop green side
    cv::Rect roi(0, 0, visionInfo.width(), visionInfo.height());
    cv::Mat srcImage = imgBgr(roi);
    
    // image_fusion
    cv::Mat dst;
    cv::addWeighted(srcImage, 1, threeChannelMask, FUSION_COEFFICIENT, 0, dst);

    // mat2vision
    ret = Mat2MxpiVision(idx, dst, dstMxpiVision);
    if (ret != APP_ERR_OK) {
        LogError << "convert mat to mxvision failed!";
        return ret;
    }
    LogInfo << " MxpiRoadSegPostProcess::openCVImageFusion end.";
    return APP_ERR_OK;
}
APP_ERROR MxpiRoadSegPostProcess::Mat2MxpiVision(size_t idx, const cv::Mat& mat, MxTools::MxpiVision& vision)
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


APP_ERROR MxpiRoadSegPostProcess::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "MxpiRoadSegPostProcess::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    // Get the property values by key 插件对应的属性值将通过“configParamMap”入参传入，可通过属性名称获取。
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get();
    return APP_ERR_OK;
}

APP_ERROR MxpiRoadSegPostProcess::DeInit()
{
    LogInfo << "MxpiRoadSegPostProcess::DeInit end.";
    return APP_ERR_OK;
}

APP_ERROR MxpiRoadSegPostProcess::SetMxpiErrorInfo(MxpiBuffer& buffer,
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

APP_ERROR MxpiRoadSegPostProcess::GenerateVisionListOutput(const MxpiTensorPackageList srcMxpiTensorPackageList,
                                                           const MxpiVisionList srcMxpiVisionList,
                                                           MxpiVisionList& dstMxpiVisionList)
{
    // Get Tensor
    std::vector<MxBase::TensorBase> tensors = {};
    GetTensors(srcMxpiTensorPackageList, tensors);
    
    // nchw_to_nhwc
    std::vector<MxBase::TensorBase> resizedTensors = {};
    APP_ERROR ret = chw2hwc(tensors, resizedTensors);
    if (ret != APP_ERR_OK) {
         LogInfo <<"chw_to_hwc failed";
    }
    
    // PostProcess
    auto srcMxpiVision = srcMxpiVisionList.visionvec(0);
    ImageInfo imageInfo;
    imageInfo.imgHeight = srcMxpiVision.visioninfo().height();
    imageInfo.imgWidth = srcMxpiVision.visioninfo().width();
    cv::Mat mask;
    ret = PostProcess(resizedTensors, imageInfo, mask);
    if (ret != APP_ERR_OK) {
        LogInfo <<"PostProcess failed";
    }
    // single channel mask to three channel mask
    cv::Mat threeChannelMask = Mat::zeros(mask.rows, mask.cols, CV_8UC3);
    vector<cv::Mat> channels;
    const int CHANNEL_NUMBER = 3;
    for (int i = 0; i < CHANNEL_NUMBER; i++)
    {
        channels.push_back(mask);
    }
    merge(channels, threeChannelMask);
    // image fusion
    ret = GenerateVisionList(threeChannelMask, srcMxpiVisionList, dstMxpiVisionList);
    if (ret != APP_ERR_OK) {
        LogInfo <<"GenerateVisionList failed";
    }
    return APP_ERR_OK;
}

APP_ERROR MxpiRoadSegPostProcess::Process(std::vector<MxpiBuffer*>& mxpiBuffer)
{
    LogInfo << "MxpiRoadSegPostProcess::Process start";
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
        LogError << "MxpiRoadSegPostProcess process is not implemented";
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
    // get the decode image from buffer
    shared_ptr<void> vision_metadata = mxpiMetadataManager.GetMetadata("mxpi_imagedecoder0");
    if (vision_metadata == nullptr) {
        ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, "mxpi_imagedecoder0") << "Vision_Metadata is NULL, failed";
        mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, "mxpi_imagedecoder0", mxpiErrorInfo);
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
    
    // check whether the proto struct name is MxpiVisionList
    google::protobuf::Message* vision_msg = (google::protobuf::Message*)vision_metadata.get();
    const google::protobuf::Descriptor* vision_desc = vision_msg->GetDescriptor();
    if (vision_desc->name() != VISION_KEY) {
        ErrorInfo_ << GetError(APP_ERR_PROTOBUF_NAME_MISMATCH, pluginName_)
                   << "Proto struct name is not MxpiVisonList, failed";
        mxpiErrorInfo.ret = APP_ERR_PROTOBUF_NAME_MISMATCH;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, "mxpi_imagedecoder0", mxpiErrorInfo);
        return APP_ERR_PROTOBUF_NAME_MISMATCH; // self define the error code
    }
    
    // Generate VisionList output
    shared_ptr<MxpiTensorPackageList> srcMxpiTensorPackageListSptr = static_pointer_cast<MxpiTensorPackageList>(metadata);
    shared_ptr<MxpiVisionList> srcMxpiMxpiVisionListSptr = static_pointer_cast<MxpiVisionList>(vision_metadata);
    shared_ptr<MxpiVisionList> dstMxpiVisionListSptr = make_shared<MxpiVisionList>();
    APP_ERROR ret = GenerateVisionListOutput(*srcMxpiTensorPackageListSptr, *srcMxpiMxpiVisionListSptr, *dstMxpiVisionListSptr);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret, pluginName_) << "MxpiRoadSegPostProcess gets inference information failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Add Generated data to metedata
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiVisionListSptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiRoadSegPostProcess add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    LogInfo << "MxpiRoadSegPostProcess::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiRoadSegPostProcess::DefineProperties()
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
MX_PLUGIN_GENERATE(MxpiRoadSegPostProcess)

