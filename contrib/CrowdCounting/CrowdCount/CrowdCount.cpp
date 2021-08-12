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
#include "opencv2/opencv.hpp"
#include "CrowdCount.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include <unistd.h>
#include <sys/stat.h>
#include <CrowdCountPostProcess.h>

namespace{
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
    const uint32_t VPC_H_ALIGN = 2;
    const uint32_t RESIZE_WIDTH = 1408;
    const uint32_t RESIZE_HEIGHT = 800;
}
// 设置配置参数
void CrowdCount::SetCrowdCountPostProcessConfig(const InitParam &initParam,
                                                       std::map<std::string, std::shared_ptr<void>> &config) {
    MxBase::ConfigData configData;
    const std::string checkTensor = initParam.checkTensor ? "true" : "false";
    configData.SetJsonValue("CHECK_MODEL", checkTensor);
    configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
    auto jsonStr = configData.GetCfgJson().serialize();
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
}

APP_ERROR CrowdCount::Init(const InitParam &initParam) {
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
    SetCrowdCountPostProcessConfig(initParam, config);
    // 初始化CrowdCount后处理对象
    post_ = std::make_shared<CrowdCountPostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "CrowdCountstProcess init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR CrowdCount::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}
// 调用mxBase::DvppWrappe.DvppJpegDecode()函数完成图像解码，VpcResize()完成缩放
APP_ERROR CrowdCount::ReadImage(const std::string &imgPath, MxBase::TensorBase &tensor) {
    MxBase::DvppDataInfo output = {};
    APP_ERROR ret = dvppWrapper_->DvppJpegDecode(imgPath, output);
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper DvppJpegDecode failed, ret=" << ret << ".";
        return ret;
    }
    // 将数据转为到DEVICE侧，以便后续处理
    MxBase::MemoryData memoryData((void*)output.data, output.dataSize,
	                            MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
    // 对解码后图像对齐尺寸进行判定
    if (output.heightStride % VPC_H_ALIGN != 0) {
        LogError << "Output data height(" << output.heightStride << ") can't be divided by " << VPC_H_ALIGN << ".";
        MxBase::MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    std::vector<uint32_t> shape = {output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, output.widthStride};
    tensor = MxBase::TensorBase(memoryData, false, shape, MxBase::TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR CrowdCount::Resize(const MxBase::TensorBase &inputTensor, MxBase::TensorBase &outputTensor){
    auto shape = inputTensor.GetShape();
    // 还原为原始尺寸
    MxBase::DvppDataInfo input = {};
    input.height = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.width = shape[1];
    input.heightStride = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.widthStride = shape[1];
    input.dataSize = inputTensor.GetByteSize();
    input.data = (uint8_t*)inputTensor.GetBuffer();
    MxBase::ResizeConfig resize = {};
    resize.height = RESIZE_HEIGHT;
    resize.width = RESIZE_WIDTH;
    MxBase::DvppDataInfo output = {};
    // 图像缩放
    APP_ERROR ret = dvppWrapper_->VpcResize(input, output, resize);
    if (ret != APP_ERR_OK) {
        LogError << "VpcResize failed, ret=" << ret << ".";
        return ret;
    }
    // 对缩放后图像对齐尺寸进行判定
    MxBase::MemoryData memoryData((void*)output.data, output.dataSize, 
	                            MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
    if (output.heightStride % VPC_H_ALIGN != 0) {
        LogError << "Output data height(" << output.heightStride << ") can't be divided by " << VPC_H_ALIGN << ".";
        MxBase::MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    shape = {output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, output.widthStride};
    outputTensor = MxBase::TensorBase(memoryData, false, shape,MxBase::TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;

}
// 模型推理
APP_ERROR CrowdCount::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                           std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
	// 人群计数模型检测特征图尺寸
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t) modelDesc_.outputTensors[i].tensorDims[j]);
        }
	// 用检测特征图尺寸分别为数据构建空的tensor
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    // 设置类型为静态batch
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_inferCost.push_back(costMs);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}
// 后处理
APP_ERROR CrowdCount::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                                 std::vector<MxBase::TensorBase> &images, std::vector<int> &results)
{
    // 使用CrowdCountPostProcess post_;
    APP_ERROR ret = post_->Process(inputs, images, results);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}
// 输出人群计数结果图
APP_ERROR CrowdCount::WriteResult(const std::vector<MxBase::TensorBase> &outputs, 
		                 const std::vector<MxBase::TensorBase> &postimage, const std::vector<int> &results) {
    cv::Mat mergeImage;
    cv::Mat dstimgBgr;
    double alpha = 1.0;
    double beta = 0.5;
    double gamma = 0.0;    
    cv::Mat imgBgr;
    imgBgr = cv::imread("crowd.jpg");
    imageWidth_ = imgBgr.cols;
    imageHeight_ = imgBgr.rows;
    int number = results[0];
    std::string str;
    str = std::to_string(number);
    float fontScale_ = 3;
    uint32_t thickness_ = 8;
    auto shape = outputs[0].GetShape();
    cv::Mat heatMap = cv::Mat(shape[1], shape[2], CV_8UC3, outputs[0].GetBuffer());
    auto image = postimage[0];
    APP_ERROR ret  = image.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "ToHost Failed";
        return ret;
    }
    shape = image.GetShape();
    cv::Mat imageYuv = cv::Mat(shape[0], shape[1], CV_8UC1,  image.GetBuffer());
    cv::cvtColor(imageYuv, dstimgBgr, cv::COLOR_YUV2BGR_NV12);
    addWeighted(dstimgBgr, alpha, heatMap, beta, gamma, mergeImage);
    // 将得到的mergeImage图片的大小还原成被测试图片的大小
    resize(mergeImage, mergeImage, cv::Size(imageWidth_, imageHeight_), 0, 0, cv::INTER_LINEAR);
    // 将计算得到的result结果写在输出结果的图片上
    cv::putText(mergeImage, str, cv::Point(30, 120),
                cv::FONT_HERSHEY_SIMPLEX, fontScale_, cv::Scalar(0,0,255), thickness_);
    cv::imwrite("./result.jpg", mergeImage);
    return APP_ERR_OK;
}

APP_ERROR CrowdCount::Process(const std::string &imgPath) {
    // read image
    MxBase::TensorBase inTensor;
    APP_ERROR ret = ReadImage(imgPath, inTensor);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    //do resize
    MxBase::TensorBase outTensor;
    ret = Resize(inTensor, outTensor);
    if (ret != APP_ERR_OK) {
        LogError << "Resize failed, ret=" << ret << ".";
        return ret;
    }
    //do inference 
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    inputs.push_back(outTensor);
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "Inference";
    //do postprocess
    std::vector<MxBase::TensorBase> heatmap = {};
    std::vector<int> results = {};
    ret = PostProcess(outputs, heatmap, results);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    //wirte result
    std::vector<MxBase::TensorBase> postimage = {outTensor};
    ret = WriteResult(heatmap, postimage, results);
    if (ret != APP_ERR_OK) {
        LogError << "Save result failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}
