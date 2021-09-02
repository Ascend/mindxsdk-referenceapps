/**
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
#include "RcfDetection.h"
#include "opencv2/opencv.hpp"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "../rcfPostProcess/RcfPostProcess.h"
#include <unistd.h>
#include <sys/stat.h>

namespace{
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
    const uint32_t VPC_H_ALIGN = 2;
    const uint32_t channel = 3;
    const double alpha1 = 255.0/255;
}
void RcfDetection::SetRcfPostProcessConfig(const InitParam &initParam,
                                           std::map<std::string, std::shared_ptr<void>> &config) {
    MxBase::ConfigData configData;
    const std::string checkTensor = initParam.checkTensor ? "true" : "false";
    
    configData.SetJsonValue("OUTSIZE_NUM", std::to_string(initParam.outSizeNum));
    configData.SetJsonValue("OUTSIZE", initParam.outSize);
    configData.SetJsonValue("RCF_TYPE", std::to_string(initParam.rcfType));
    configData.SetJsonValue("MODEL_TYPE", std::to_string(initParam.modelType));
    configData.SetJsonValue("INPUT_TYPE", std::to_string(initParam.inputType));
    configData.SetJsonValue("CHECK_MODEL", checkTensor);
    auto jsonStr = configData.GetCfgJson().serialize();
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
}

APP_ERROR RcfDetection::Init(const InitParam &initParam) {
    deviceId_ = initParam.deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }
    dvppWrapper_ = std::make_shared<MxBase::DvppWrapper>();
    ret = dvppWrapper_->Init();
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper init failed, ret=" << ret << ".";
        return ret;
    }
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    std::map<std::string, std::shared_ptr<void>> config;
    SetRcfPostProcessConfig(initParam, config);
    post_ = std::make_shared<RcfPostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "RcfPostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    if (ret != APP_ERR_OK) {
        LogError << "Failed to load labels, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR RcfDetection::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR RcfDetection::ReadImage(const std::string &imgPath,
	                                MxBase::TensorBase &tensor) {
    MxBase::DvppDataInfo output = {};
    APP_ERROR ret = dvppWrapper_->DvppJpegDecode(imgPath, output);
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper DvppJpegDecode failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::MemoryData memoryData((void*)output.data, output.dataSize, 
		                             MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
    if (output.heightStride % VPC_H_ALIGN != 0) {
        LogError << "Output data height(" << output.heightStride << ") can't be divided by " << VPC_H_ALIGN << ".";
        MxBase::MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    dvppHeightStride = output.heightStride;
    dvppWidthStride = output.widthStride;
    std::vector<uint32_t> shape = {output.heightStride * YUV_BYTE_NU/YUV_BYTE_DE , output.widthStride};
    tensor = MxBase::TensorBase(memoryData, false, shape, MxBase::TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}
APP_ERROR RcfDetection::Resize(const MxBase::TensorBase &inputTensor, MxBase::TensorBase &outputTensor,
	                              uint32_t resizeHeight, uint32_t resizeWidth) {
    auto shape = inputTensor.GetShape();
    MxBase::DvppDataInfo input = {};
    input.height = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.width = shape[1];
    input.heightStride = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.widthStride = shape[1];
    input.dataSize = inputTensor.GetByteSize();
    input.data = (uint8_t*)inputTensor.GetBuffer();
    MxBase::ResizeConfig resize = {};
    resize.height = resizeHeight;
    resize.width = resizeWidth;
    MxBase::DvppDataInfo output = {};
    APP_ERROR ret = dvppWrapper_->VpcResize(input, output, resize);
    if (ret != APP_ERR_OK) {
        LogError << "VpcResize failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::MemoryData memoryData((void*)output.data, output.dataSize,
		                              MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
    if (output.heightStride % VPC_H_ALIGN != 0) {
        LogError << "Output data height(" << output.heightStride << ") can't be divided by " << VPC_H_ALIGN << ".";
        MxBase::MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    shape = {1, channel, output.heightStride, output.widthStride};
    outputTensor = MxBase::TensorBase(memoryData, false, shape,MxBase::TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR RcfDetection::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                  std::vector<MxBase::TensorBase> &outputs) {
    std::vector<MxBase::TensorBase> output={};
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    LogInfo<< "costMs:"<< costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR RcfDetection::PostProcess(const MxBase::TensorBase &tensor,
		                                const std::vector<MxBase::TensorBase> &outputs,
						                        std::vector<MxBase::TensorBase> &postProcessOutput) {
    auto shape = tensor.GetShape();
    MxBase::ResizedImageInfo imgInfo;
    imgInfo.widthOriginal = shape[1];
    imgInfo.heightOriginal = shape[0] * YUV_BYTE_DE;
    imgInfo.widthResize = 512;
    imgInfo.heightResize = 512;
    imgInfo.resizeType = MxBase::RESIZER_STRETCHING;
    std::vector<MxBase::ResizedImageInfo> imageInfoVec = {};
    imageInfoVec.push_back(imgInfo);
    APP_ERROR ret = post_->Process(outputs, postProcessOutput);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    ret = post_->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "RcfDetection DeInit failed";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR RcfDetection::WriteResult(MxBase::TensorBase &inferTensor,const std::string &imgPath) {
    auto shape = inferTensor.GetShape();
    uint32_t height = shape[2];
    uint32_t width = shape[3];
    cv::Mat imgBgr = cv::imread(imgPath);
    uint32_t imageWidth = imgBgr.cols;
    uint32_t imageHeight = imgBgr.rows;
    cv::Mat modelOutput = cv::Mat(height, width, CV_32FC1, inferTensor.GetBuffer());
    cv::Mat grayMat;
    cv::Mat resizedMat;
    int crop = 5;
    cv::Rect myROI(0, 0, imageWidth-crop, imageHeight );
    resize(modelOutput, resizedMat, cv::Size(dvppWidthStride, dvppHeightStride), 0, 0, cv::INTER_LINEAR);
    resizedMat.convertTo(grayMat, CV_8UC1, alpha1);
    cv::Mat croppedImage = grayMat(myROI);
    resize(croppedImage, croppedImage,  cv::Size(imageWidth, imageHeight), 0, 0, cv::INTER_LINEAR);
    cv::imwrite("./result.jpg", croppedImage);
    return APP_ERR_OK;
}

APP_ERROR RcfDetection::Process(const std::string &imgPath) {
    MxBase::TensorBase inTensor;
    APP_ERROR ret = ReadImage(imgPath, inTensor);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::TensorBase outTensor;
    uint32_t resizeHeight = 512;
    uint32_t resizeWidth = 512;
    ret = Resize(inTensor, outTensor, resizeHeight, resizeWidth);
    if (ret != APP_ERR_OK) {
        LogError << "Resize failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    auto shape = outTensor.GetShape();
    inputs.push_back(outTensor);
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<MxBase::TensorBase> postProcessOutput={};
    ret = PostProcess(inTensor, outputs, postProcessOutput);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    ret = WriteResult(postProcessOutput[0], imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "Save result failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}
