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

#include "MxpiPNetPreprocess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"


using namespace MxBase;
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
namespace {
    const string VISION_KEY = "MxpiVisionList";
    const mindxsdk_private::protobuf::uint32 MXBASE_PIXEL_FORMAT_YUV_SEMIPLANAR_420 = 1;
    const mindxsdk_private::protobuf::uint32 MXBASE_PIXEL_FORMAT_RGB_888 = 12;
    const mindxsdk_private::protobuf::uint32 MXBASE_PIXEL_FORMAT_BGR_888 = 13;
    const int YUV_U = 2;
    const int YUV_V = 3;
    const int ER = 2;
    const int ER_WU_LIU = 256
    const float LING_DIAN_WU = 0.5
}


APP_ERROR MxpiPNetPreprocess::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap) {
    LogInfo << "MxpiPoseEstNetPrepostPlugin::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    // Get the property values by key
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get();
    std::shared_ptr<string> outputFormatPropSptr = std::static_pointer_cast<string>(configParamMap["outputFormat"]);
    outputDataFormat = *outputFormatPropSptr.get();
    std::shared_ptr<string> descriptionMessageProSptr = std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
    descriptionMessage_ = *descriptionMessageProSptr.get();
    LogInfo << "MxpiPoseEstNetPrepostPlugin::Init complete.";
    return APP_ERR_OK;
}


APP_ERROR MxpiPNetPreprocess::DeInit() {
    LogInfo << "MxpiPNetPreprocess::DeInit start.";
    LogInfo << "MxpiPNetPreprocess::DeInit complete.";
    return APP_ERR_OK;
}


APP_ERROR MxpiPNetPreprocess::SetMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,
                                               const MxpiErrorInfo mxpiErrorInfo) {
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


APP_ERROR MxpiPNetPreprocess::GenerateVisionList(const MxpiVisionList srcMxpiVisionList,
                                                 MxpiVisionList& dstMxpiVisionList) {
    LogInfo << "input type:" << srcMxpiVisionList.visionvec(0).visiondata().datatype();
    for (int i = 0; i< srcMxpiVisionList.visionvec_size();i++) {
        auto srcMxpiVision = srcMxpiVisionList.visionvec(i);
        MxTools::MxpiVision dstVision;
        APP_ERROR ret = DoAffineTransform(i, srcMxpiVision, dstVision);
        if (ret != APP_ERR_OK) {
            LogWarn << "element(" << elementName_ << ") normailze failed";
        }
        dstMxpiVisionList.add_visionvec()->CopyFrom(dstVision);
    }
    if (dstMxpiVisionList.visionvec_size() == 0) {
        LogError <<  "element(" << elementName_ << ") dst vision vec size is 0!";
        return APP_ERR_COMM_FAILURE;
    }
    LogInfo << "Generate done";
    return APP_ERR_OK;
}


APP_ERROR MxpiPNetPreprocess::DoAffineTransform(size_t idx,
                                                const MxTools::MxpiVision srcMxpiVision,
                                                MxTools::MxpiVision& dstMxpiVision) {
    LogInfo << "Till now is ok";
    auto &visionInfo = srcMxpiVision.visioninfo();
    auto &visionData = srcMxpiVision.visiondata();

    MemoryData memorySrc = {};
    memorySrc.deviceId = visionData.deviceid();
    memorySrc.type = (MxBase::MemoryData::MemoryType)visionData.memtype();
    memorySrc.size = visionData.datasize();
    memorySrc.ptrData = (void *)visionData.dataptr();
    MemoryData memoryDst(visionData.datasize(), MemoryData::MEMORY_HOST_NEW);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst, memorySrc);
    if (ret != APP_ERR_OK) {
        LogError << "Fail to malloc and copy host memory.";
        return ret;
    }

    int orignal_width;
    int orignal_height;

    cv::Mat src;
    cv::Mat imgBGR;
    cv::Mat imgRGB;

    // check the format of the input image, decode it accordingly.
    outputPixelFormat_ = visionInfo.format();
    LogInfo << outputPixelFormat_;
    if (outputPixelFormat_ == MXBASE_PIXEL_FORMAT_YUV_SEMIPLANAR_420) {
        LogInfo << "the format of input image is YUV, using dvpp to decode.";
        imgRGB = cv::Mat(visionInfo.height(), visionInfo.width(), CV_8UC3);
        orignal_width = visionInfo.width();
        orignal_height = visionInfo.height();

        cv::Mat transedImgRBG = cv::Mat(visionInfo.heightaligned(), visionInfo.widthaligned(), CV_8UC3);
        src = cv::Mat(visionInfo.heightaligned()* YUV_V / YUV_U, visionInfo.widthaligned(), CV_8UC1, memoryDst.ptrData);
        cv::cvtColor(src, transedImgRBG, cv::COLOR_YUV2RGB_NV12);

        vector<int> cropRange;
        cropRange.push_back(0); // x_0
        cropRange.push_back(0); // y_0
        cropRange.push_back(orignal_width); // x_1
        cropRange.push_back(orignal_height); // y_1
        crop_img(transedImgRBG, imgRGB, cropRange);
    } else {
        imgRGB = cv::Mat(visionInfo.heightaligned(), visionInfo.widthaligned(), CV_8UC3);
        orignal_width = visionInfo.widthaligned();
        orignal_height = visionInfo.heightaligned();
        if (outputPixelFormat_ == MXBASE_PIXEL_FORMAT_RGB_888) {
            LogInfo << "the format of input image is RGB, using opencv to decode.";
            imgRGB = cv::Mat(visionInfo.heightaligned(), visionInfo.widthaligned(), CV_8UC3, memoryDst.ptrData);
        } else if (outputPixelFormat_ == MXBASE_PIXEL_FORMAT_BGR_888) {
            LogInfo << "the format of input image is BGR, using opencv to decode.";
            src = cv::Mat(visionInfo.heightaligned(), visionInfo.widthaligned(), CV_8UC3, memoryDst.ptrData);
            cv::cvtColor(src, imgRGB, cv::COLOR_BGR2RGB);
        }
    }

    cv::Mat dst;
    MxBase::MemoryData memoryNewDst(dst.data, MxBase::MemoryData::MEMORY_HOST_NEW);
    auto pixel_std = 200;
    nc::NdArray<float> center = {float(orignal_width / 2.0), float(orignal_height / 2.0)};
    float scale = MAX(orignal_width, orignal_height) * 1.25 / pixel_std;
    int outputSize[2] = { ER_WU_LIU, ER_WU_LIU };
    cv::Size dstSize = { ER_WU_LIU, ER_WU_LIU };
    cv::Mat trans = get_affine_transform(center, scale, 0, outputSize, {0, 0}, 0);
    cv::warpAffine(imgRGB, dst, trans, dstSize);

    cv::Mat imgYuv;
    cv::Mat yuv_mat;
    cv::Mat img_nv12;
    if (outputDataFormat == "YUV") {
        LogInfo << "output in yuv";
        imgYuv = cv::Mat(ER_WU_LIU, ER_WU_LIU, CV_8UC1);
        cv::cvtColor(dst, imgYuv, cv::COLOR_RGB2YUV_I420);
        yuv_mat = imgYuv.clone();
        Bgr2Yuv(yuv_mat, img_nv12);
        ret = Mat2MxpiVisionDvpp(idx, img_nv12, dstMxpiVision);
    } else {
        if (outputDataFormat == "RGB") {
            LogInfo << "output in rgb";
            ret = Mat2MxpiVisionOpencv(idx, dst, dstMxpiVision);
        } else {
            LogInfo << "output in bgr";
            imgBGR = cv::Mat(ER_WU_LIU, ER_WU_LIU, CV_8UC3);
            cv::cvtColor(dst, imgBGR, cv::COLOR_RGB2BGR);
            ret = Mat2MxpiVisionOpencv(idx, imgBGR, dstMxpiVision);
        }
    }

    if (ret != APP_ERR_OK) {
        LogError << "convert mat to mxvision failed!";
        return ret;
    }
    LogInfo << "affine_transform done";
    return APP_ERR_OK;
}


APP_ERROR MxpiPNetPreprocess::Mat2MxpiVisionOpencv(size_t idx, const cv::Mat& mat, MxTools::MxpiVision& vision) {
    LogInfo << "Mat2MxpiVision begin";
    auto header = vision.add_headervec();
    header->set_memberid(idx);
    header->set_datasource(parentName_);

    auto visionInfo = vision.mutable_visioninfo();
    if (outputDataFormat == "YUV") {
        visionInfo->set_format(MXBASE_PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    } else {
        if (outputDataFormat == "RGB") {
            visionInfo->set_format(MXBASE_PIXEL_FORMAT_RGB_888);
        } else {
            visionInfo->set_format(MXBASE_PIXEL_FORMAT_BGR_888);
        }
    }

    visionInfo->set_height(mat.rows);
    visionInfo->set_heightaligned(mat.rows);
    visionInfo->set_width(mat.cols);
    visionInfo->set_widthaligned(mat.cols);
    auto visionData = vision.mutable_visiondata();
    LogInfo << "elemSize = " << mat.elemSize();
    LogInfo << "col = " << mat.cols;
    LogInfo << "rows = " << mat.rows;
    LogInfo << "size = " << mat.size();
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
    visionData->set_datatype(MxTools::MxpiDataType::MXPI_DATA_TYPE_UINT8);
    LogInfo << "Mat2MxpiVision done";
    return APP_ERR_OK;
}


// area is the upper left corner coordinate and width height of the cutting area
void MxpiPNetPreprocess::crop_img(cv::Mat &img, cv::Mat &crop_img, std::vector<int> &area) {
    int crop_x1 = std::max(0, area[0]);
    int crop_y1 = std::max(0, area[1]);
    int crop_x2 = std::min(img.cols -1, area[0] + area[2] - 1);
    int crop_y2 = std::min(img.rows - 1, area[1] + area[3] - 1);

    crop_img = img(cv::Range(crop_y1, crop_y2+1), cv::Range(crop_x1, crop_x2 + 1));
}


APP_ERROR MxpiPNetPreprocess::Bgr2Yuv(cv::Mat &yuv_mat, cv::Mat &img_nv12) {
    uint8_t *yuv = yuv_mat.ptr<uint8_t>();
    img_nv12 = cv::Mat(ER_WU_LIU * YUV_V / YUV_U, ER_WU_LIU, CV_8UC1);
    uint8_t *ynv12 = img_nv12.ptr<uint8_t>();
    int32_t uv_height = 256 / 2;
    int32_t uv_width = 256 / 2;
    int32_t y_size = 256 * 256;
    memcpy(ynv12, yuv, y_size);
    uint8_t *nv12 = ynv12 + y_size;
    uint8_t *u_data = yuv + y_size;
    uint8_t *v_data = u_data + uv_height * uv_width;
    for (int32_t i = 0; i < uv_width * uv_height; i++) {
        *nv12++ = *u_data++;
        *nv12++ = *v_data++;
    }
    return APP_ERR_OK;
}


APP_ERROR MxpiPNetPreprocess::Mat2MxpiVisionDvpp(size_t idx, const cv::Mat& mat, MxTools::MxpiVision& vision) {
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
    LogInfo << "elemSize = " << mat.elemSize();
    LogInfo << "col = " << mat.cols;
    LogInfo << "rows = " << mat.rows;
    LogInfo << "size = " << mat.size();
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
    visionData->set_datatype(MxTools::MxpiDataType::MXPI_DATA_TYPE_UINT8);
    LogInfo << "Mat2MxpiVision done";
    return APP_ERR_OK;
};


cv::Mat MxpiPNetPreprocess::get_affine_transform(nc::NdArray<float> center, float scale, float rot,
                                                 int output_size[2], nc::NdArray<float> shift = {0, 0}, int inv = 0) {
    nc::NdArray<float> scales = { scale * 200, scale * 200 };
    nc::NdArray<float> scale_tmp = scales;
    float src_w = scale_tmp[0];
    int dst_w = output_size[0];
    int dst_h = output_size[1];
    float rot_rad = nc::constants::pi * rot / 180;
    nc::NdArray<float> src_dir = get_dir(nc::NdArray<float> { 0, float(src_w * -0.5) }, rot_rad);
    nc::NdArray<float> dst_dir = { 0, float(dst_w * -0.5) };
    nc::NdArray<float> src = nc::zeros<float>(3, 2);
    nc::NdArray<float> dst = nc::zeros<float>(3, 2);
    nc::NdArray<float> temp;

    temp = center + scale_tmp * shift;
    src(0, 0) = temp(0, 0);
    src(0, 1) = temp(0, 1);
    temp = center + src_dir + scale_tmp * shift;
    src(1, 0) = temp(0, 0);
    src(1, 1) = temp(0, 1);
    temp = dst_w * LING_DIAN_WU, dst_h * LING_DIAN_WU;
    dst(0, 0) = temp(0, 0);
    dst(0, 1) = temp(0, 1);
    temp = nc::NdArray<float> {float(dst_w * LING_DIAN_WU), float(dst_h * LING_DIAN_WU)} + dst_dir;
    dst(1, 0) = temp(0, 0);
    dst(1, 1) = temp(0, 1);
    temp = get_3rd_point(src(0, src.cSlice()), src(1, src.cSlice()));
    src(ER, 0) = temp(0, 0);
    src(ER, 1) = temp(0, 1);
    temp = get_3rd_point(dst(0, dst.cSlice()), dst(1, dst.cSlice()));
    dst(ER, 0) = temp(0, 0);
    dst(ER, 1) = temp(0, 1);

    cv::Mat trans;
    cv::Point2f SRC[3];
    cv::Point2f DST[3];
    SRC[0] = cv::Point2f(src(0, 0), src(0, 1));
    SRC[1] = cv::Point2f(src(1, 0), src(1, 1));
    SRC[ER] = cv::Point2f(src(ER, 0), src(ER, 1));
    DST[0] = cv::Point2f(dst(0, 0), dst(0, 1));
    DST[1] = cv::Point2f(dst(1, 0), dst(1, 1));
    DST[ER] = cv::Point2f(dst(ER, 0), dst(ER, 1));
    if (1 == inv) {
        trans = cv::getAffineTransform(DST, SRC);
    }
    else
    {
        trans = cv::getAffineTransform(SRC, DST);
    }
    return trans;
}


nc::NdArray<float> MxpiPNetPreprocess::get_dir(nc::NdArray<float> src_point, float rot_rad) {
    float sn = nc::sin(rot_rad);
    float cs = nc::cos(rot_rad);
    nc::NdArray<float> src_result = {0, 0};
    src_result[0] = src_point[0] * cs - src_point[1] * sn;
    src_result[1] = src_point[0] * sn + src_point[1] * cs;
    return src_result;
}


nc::NdArray<float> MxpiPNetPreprocess::get_3rd_point(nc::NdArray<float> a, nc::NdArray<float> b) {
    nc::NdArray<float> direct = a - b;
    nc::NdArray<float> c  = b + nc::NdArray<float> { -direct[1], direct[0] };
    return c;
}


APP_ERROR MxpiPNetPreprocess::Process(std::vector<MxpiBuffer*>& mxpiBuffer) {
    LogInfo << "MxpiPNetPreprocess::Process start";
    MxpiBuffer* buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) << "MxpiPNetPreprocess process is not implemented";
        mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        LogError << "MxpiPNetPreprocess process is not implemented";
        return APP_ERR_COMM_FAILURE;
    }
    // Get the (image) data from buffer
    shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(parentName_);
    if (metadata == nullptr) {
        ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "Metadata of tensor is NULL, failed";
        mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_METADATA_IS_NULL; // self define the error code
    }
    // Check the proto struct name
    google::protobuf::Message* msg = (google::protobuf::Message*)metadata.get();
    const google::protobuf::Descriptor* desc = msg->GetDescriptor();
    if (desc->name() != VISION_KEY) {
        ErrorInfo_ << GetError(APP_ERR_PROTOBUF_NAME_MISMATCH, pluginName_)
        << "Proto struct name is not MxpiVisionList, failed";
        mxpiErrorInfo.ret = APP_ERR_PROTOBUF_NAME_MISMATCH;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_PROTOBUF_NAME_MISMATCH; // self define the error code
    }
    LogInfo << "MxpiPNetPreprocess::Get Original Image Completed";

    // Generate WHENet output
    shared_ptr<MxpiVisionList> srcMxpiVisionListSptr = static_pointer_cast<MxpiVisionList>(metadata);
    shared_ptr<MxpiVisionList> dstMxpiVisionListSptr = make_shared<MxpiVisionList>();
    APP_ERROR ret = GenerateVisionList(*srcMxpiVisionListSptr, *dstMxpiVisionListSptr);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret, pluginName_) << "MxpiPNetPreprocess gets inference information failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }

    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiVisionListSptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiPNetPreprocess add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    LogInfo << "MxpiPNetPreprocess::Process end";
    return APP_ERR_OK;
}


std::vector<std::shared_ptr<void>> MxpiPNetPreprocess::DefineProperties() {
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto parentNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "dataSource", "name", "the name of previous plugin", "mxpi_imagedecoder0", "NULL", "NULL"});
    auto outputPixelFormatProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "outputFormat", "name", "the format of output image", "YUV", "NULL", "NULL"});
    auto descriptionMessageProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "descriptionMessage", "message", "Description mesasge of plugin", "This is MxpiSamplePlugin", "NULL", "NULL"});
    properties.push_back(parentNameProSptr);
    properties.push_back(outputPixelFormatProSptr);
    properties.push_back(descriptionMessageProSptr);

    return properties;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiPNetPreprocess)