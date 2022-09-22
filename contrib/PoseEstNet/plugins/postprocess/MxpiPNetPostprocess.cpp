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

#include "MxpiPNetPostprocess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"


using namespace MxBase;
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
namespace {
    const string SAMPLE_KEY = "MxpiTensorPackageList";
    const string INFO_KEY = "MxpiVisionList";
    const int BIN_WIDTH_IN_DEGREES = 3;
    const int ER = 2;
    const int SAN = 3;
    const float LING_DIAN_ER_WU = 0.25;
    const float LING_DIAN_WU = 0.5;
}

// Unpack MxpiTensorPackageList
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

APP_ERROR MxpiPNetPostprocess::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap) {
    LogInfo << "MxpiPNetPostprocess::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    // Get the property values by key
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get();
    std::shared_ptr<string> infoPlugPropSptr = std::static_pointer_cast<string>(configParamMap["InfoSource"]);
    infoPlugName_ = *infoPlugPropSptr.get();
    std::shared_ptr<string> descriptionMessageProSptr =
            std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
    descriptionMessage_ = *descriptionMessageProSptr.get();
    LogInfo << "MxpiPNetPostprocess::Init complete.";
    return APP_ERR_OK;
}

APP_ERROR MxpiPNetPostprocess::DeInit() {
    LogInfo << "MxpiPNetPostprocess::DeInit start.";
    LogInfo << "MxpiPNetPostprocess::DeInit complete.";
    return APP_ERR_OK;
}

APP_ERROR MxpiPNetPostprocess::SetMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,
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

APP_ERROR MxpiPNetPostprocess::GenerateHeadPoseInfo(const MxpiTensorPackageList srcMxpiTensorPackage,
                                                    const MxpiVisionList srcMxpiVisionPackage,
                                                    MxpiObjectList& dstMxpiObjectListSptr) {
    // Get Tensors
    std::vector<MxBase::TensorBase> tensors = {};
    GetTensors(srcMxpiTensorPackage, tensors);

    auto tensor_size = tensors.size();
    for (int i = 0; i < tensor_size; i++) {
        auto dataPtr = (float *)tensors[i].GetBuffer();
        auto tensor_shape = tensors[i].GetShape();

        auto orignal_width = srcMxpiVisionPackage.visionvec()[i].visioninfo().width();
        auto orignal_height = srcMxpiVisionPackage.visionvec()[i].visioninfo().height();
        int keypoint_nums, hm_h, hm_w;
        keypoint_nums = tensor_shape[1];
        hm_h = tensor_shape[ER];
        hm_w = tensor_shape[SAN];

        nc::NdArray<float32_t> coordinate;
        coordinate = nc::zeros<float32_t>(keypoint_nums, ER);
        nc::NdArray<float32_t> maxval;
        maxval = nc::zeros<float32_t>(keypoint_nums, 1);

        for (auto j = 0; j < keypoint_nums; j++) {
            nc::NdArray<float32_t> heatmap;
            heatmap = nc::zeros<float32_t>(1, hm_h*hm_w);
            for (auto k = 0; k < hm_h*hm_w; k++) {
                heatmap(0, k) = dataPtr[j*hm_h*hm_w + k];
            }
            auto index = nc::argmax(heatmap);
            auto max_value = nc::max(heatmap);
            auto hm_x = index(0, 0) % hm_w;
            auto hm_y = floor(index(0, 0) / hm_w);
            coordinate(j, 0) = hm_x;
            coordinate(j, 1) = hm_y;
            maxval(j, 0) = max_value(0, 0);

            auto px = floor(coordinate(j, 0) + 0.5);
            auto py = floor(coordinate(j, 1) + 0.5);
            heatmap = heatmap.reshape(hm_h, hm_w);

            if ((1 < px < hm_w - 1) and (1 < py < hm_h - 1)) {
                auto diff_x = heatmap(py, px + 1) - heatmap(py, px - 1);
                auto diff_y = heatmap(py + 1, px) - heatmap(py - 1, px);
                if (diff_x > 0) {
                    coordinate(j, 0) = coordinate(j, 0) + LING_DIAN_ER_WU;
                } else if (diff_x < 0) {
                    coordinate(j, 0) = coordinate(j, 0) - LING_DIAN_ER_WU;
                } else {
                    coordinate(j, 0) = coordinate(j, 0);
                }
                if (diff_y > 0) {
                    coordinate(j, 1) = coordinate(j, 1) + LING_DIAN_ER_WU;
                } else if (diff_y < 0) {
                    coordinate(j, 1) = coordinate(j, 1) - LING_DIAN_ER_WU;
                } else {
                    coordinate(j, 1) = coordinate(j, 1);
                }
            }
        }

        auto pixel_std = 200;
        nc::NdArray<float> center = {float(orignal_width / 2.0), float(orignal_height / 2.0)};
        float scale = MAX(orignal_width, orignal_height) * 1.25 / pixel_std;

        int w_h[2] = { hm_w, hm_h };
        nc::NdArray<float> final_coordinate = transform_preds(coordinate, center, scale, w_h);

        for (auto it = 0; it < keypoint_nums; it++) {
            auto dstMxpiHObjectInfoPtr = dstMxpiObjectListSptr.add_objectvec();
            MxpiMetaHeader* dstMxpiMetaHeaderList = dstMxpiHObjectInfoPtr->add_headervec();
            dstMxpiMetaHeaderList->set_datasource(parentName_);
            dstMxpiMetaHeaderList->set_memberid(0);
            dstMxpiHObjectInfoPtr->set_x0(final_coordinate(it, 0));
            dstMxpiHObjectInfoPtr->set_y0(final_coordinate(it, 1));
            dstMxpiHObjectInfoPtr->set_x1(maxval(it, 0));
            dstMxpiHObjectInfoPtr->set_y1(0);
        }
    }
    return APP_ERR_OK;
}


nc::NdArray<float> MxpiPNetPostprocess::transform_preds(nc::NdArray<float> coords, nc::NdArray<float> center,
                                                        float scale, int output_size[2]) {
    nc::NdArray<float> target_coords = nc::zeros<float>(coords.shape());
    nc::NdArray<float> target_coords_temp;

    cv::Mat trans = get_affine_transform(center, scale, 0, output_size, {0, 0}, 1);
    nc::NdArray<float> trans_NdArray = nc::zeros<float>(trans.rows, trans.cols);
    double* ptr_data = (double*)trans.data;
    for (int i = 0; i < trans.rows; i++) {
        for (int j = 0; j < trans.cols; j++) {
            trans_NdArray(i, j) = (float)ptr_data[i * trans.cols + j];
        }
    }
    for (int p = 0; p < coords.shape().rows; p++) {
        target_coords_temp = nc::copy(affine_transform(coords(p, {0, 2}), trans_NdArray));
        for (int q = 0; q < ER; q++) {
            target_coords(p, q) = target_coords_temp(q, 0);
        }
    }
    return target_coords;
}

nc::NdArray<float> MxpiPNetPostprocess::affine_transform(nc::NdArray<float> pt, nc::NdArray<float> t) {
    nc::NdArray<float> new_pt = {pt(0, 0), pt(0, 1), 1.0};
    new_pt = new_pt.transpose();
    nc::NdArray<float> new_pt_dot = nc::dot(t, new_pt);
    nc::NdArray<float> my_pt_dot = new_pt_dot({0, 2}, 0);
    return my_pt_dot;
}

cv::Mat MxpiPNetPostprocess::get_affine_transform(nc::NdArray<float> center, float scale, float rot,
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
    temp = nc::NdArray<float> {float(dst_w * 0.5), float(dst_h * 0.5)} + dst_dir;
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

nc::NdArray<float> MxpiPNetPostprocess::get_dir(nc::NdArray<float> src_point, float rot_rad) {
    float sn = nc::sin(rot_rad);
    float cs = nc::cos(rot_rad);
    nc::NdArray<float> src_result = {0, 0};
    src_result[0] = src_point[0] * cs - src_point[1] * sn;
    src_result[1] = src_point[0] * sn + src_point[1] * cs;
    return src_result;
}

nc::NdArray<float> MxpiPNetPostprocess::get_3rd_point(nc::NdArray<float> a, nc::NdArray<float> b) {
    nc::NdArray<float> direct = a - b;
    nc::NdArray<float> c  = b + nc::NdArray<float> { -direct[1], direct[0] };
    return c;
}

APP_ERROR MxpiPNetPostprocess::Process(std::vector<MxpiBuffer*>& mxpiBuffer) {
    LogInfo << "MxpiPNetPostprocess::Process start";
    MxpiBuffer* buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        LogError << "MxpiPNetPostprocess process is not implemented";
        return APP_ERR_COMM_FAILURE;
    }
    // Get the data (infer tensor) from buffer
    shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(parentName_);
    if (metadata == nullptr) {
        LogError << "Metadata of tensor is NULL, failed";
        return APP_ERR_METADATA_IS_NULL; // self define the error code
    }
    // Check the proto struct name
    google::protobuf::Message* msg = (google::protobuf::Message*)metadata.get();
    const google::protobuf::Descriptor* desc = msg->GetDescriptor();
    if (desc->name() != SAMPLE_KEY) {
        LogError << "Proto struct name is not MxpiTensorPackageList, failed";
        return APP_ERR_PROTOBUF_NAME_MISMATCH; // self define the error code
    }

    // Get the data (image information) from buffer
    shared_ptr<void> Info_metadata = mxpiMetadataManager.GetMetadata(infoPlugName_);
    if (Info_metadata == nullptr) {
        LogError << "Metadata of original image is NULL, failed";
        return APP_ERR_METADATA_IS_NULL; // self define the error code
    }
    // Check the proto struct name
    google::protobuf::Message* info_msg = (google::protobuf::Message*)Info_metadata.get();
    const google::protobuf::Descriptor* info_desc = info_msg->GetDescriptor();
    if (info_desc->name() != INFO_KEY) {
        LogError << "Proto struct name is not MxpiTensorPackageList, failed";
        return APP_ERR_PROTOBUF_NAME_MISMATCH; // self define the error code
    }
    LogInfo << "MxpiPNetPostprocess::Get Image Info Completed";

    // Generate WHENet output
    shared_ptr<MxpiTensorPackageList> srcMxpiTensorPackageListSptr = static_pointer_cast<MxpiTensorPackageList>(metadata);
    shared_ptr<MxpiVisionList> srcMxpiVisionListSptr = static_pointer_cast<MxpiVisionList>(Info_metadata);
    shared_ptr<MxpiObjectList> dstMxpiObjectListSptr = make_shared<MxpiObjectList>();
    APP_ERROR ret = GenerateHeadPoseInfo(*srcMxpiTensorPackageListSptr, *srcMxpiVisionListSptr, *dstMxpiObjectListSptr);
    if (ret != APP_ERR_OK) {
        LogError << "MxpiPNetPostprocess gets inference information failed.";
        return ret;
    }

    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiObjectListSptr));
    if (ret != APP_ERR_OK) {
        LogError << "MxpiPNetPostprocess add metadata failed.";
        return ret;
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    LogError << "MxpiPNetPostprocess::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiPNetPostprocess::DefineProperties() {
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto parentNameProSptr = std::make_shared<ElementProperty<string>> (ElementProperty<string> {
        STRING, "dataSource", "name", "the name of previous plugin", "mxpi_tensorinfer2", "NULL", "NULL"});
    auto infoPlugProSptr = std::make_shared<ElementProperty<string>> (ElementProperty<string> {
        STRING, "InfoSource", "name", "the name of needed decoder/crop plugin", "mxpi_imagedecoder0", "NULL", "NULL"});
    auto descriptionMessageProSptr = std::make_shared<ElementProperty<string>> (ElementProperty<string> {
        STRING, "descriptionMessage", "message", "Description mesasge of plugin", "This is MxpiSamplePlugin", "NULL", "NULL"});
    properties.push_back(parentNameProSptr);
    properties.push_back(infoPlugProSptr);
    properties.push_back(descriptionMessageProSptr);
    return properties;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiPNetPostprocess)