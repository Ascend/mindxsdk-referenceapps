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

#include <sys/stat.h>
#include "CartoonGANPicture.h"
#include "MxBase/DeviceManager/DeviceManager.h"

namespace{
const uint32_t YUV_BYTE_NU = 3;
const uint32_t YUV_BYTE_DE = 2;
const uint32_t VPC_H_ALIGN = 2;
const uint32_t MODEL_INPUT_SIZE = 256;
const uint32_t DE_MORMALIZE = 1;
const double_t COVER_TO_COLOR = 127.5;
}

APP_ERROR CartoonGANPicture::Init(const InitParam &initParam)
{
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

    return APP_ERR_OK;
}

APP_ERROR CartoonGANPicture::DeInit()
{
    dvppWrapper_->DeInit();
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR CartoonGANPicture::ReadImage(const std::string &imgPath, MxBase::TensorBase &tensor)
{
    MxBase::DvppDataInfo output = {};
    // 图像解码
    APP_ERROR ret = dvppWrapper_->DvppJpegDecode(imgPath, output);
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper DvppJpegDecode failed, ret=" << ret << ".";
        return ret;
    }
    // 将数据转为到DEVICE侧，以便后续处理
    MxBase::MemoryData memoryData((void*)output.data, output.dataSize,
                                  MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
    imageWidth_ = output.width;
    imageHeight_ = output.height;
    widthStride_ = output.widthStride;
    heightStride_ = output.heightStride;

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

APP_ERROR CartoonGANPicture::Resize(const MxBase::TensorBase &inputTensor, MxBase::TensorBase &outputTensor)
{
    auto shape = inputTensor.GetShape();
    MxBase::DvppDataInfo input = {};
    // 还原为原始尺寸
    input.height = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.width = shape[1];
    input.heightStride = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.widthStride = shape[1];
    input.dataSize = inputTensor.GetByteSize();
    input.data = (uint8_t*)inputTensor.GetBuffer();
    MxBase::ResizeConfig resize = {};
    resize.height = MODEL_INPUT_SIZE;
    resize.width = MODEL_INPUT_SIZE;
    MxBase::DvppDataInfo output = {};

    // 图像缩放
    APP_ERROR ret = dvppWrapper_->VpcResize(input, output, resize);
    if (ret != APP_ERR_OK) {
        LogError << "VpcResize failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::MemoryData memoryData((void*)output.data, output.dataSize,
                                  MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
    // 对缩放后图像对齐尺寸进行判定
    if (output.heightStride % VPC_H_ALIGN != 0) {
        LogError << "Output data height(" << output.heightStride << ") can't be divided by " << VPC_H_ALIGN << ".";
        MxBase::MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    shape = {output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, output.widthStride};
    outputTensor = MxBase::TensorBase(memoryData, false, shape,MxBase::TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR CartoonGANPicture::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                       std::vector<MxBase::TensorBase> &outputs)
{
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
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR CartoonGANPicture::PostProcess(std::vector<MxBase::TensorBase> &outputs, cv::Mat &result)
{
    for(uint32_t i = 0; i < outputs.size(); i++){
        APP_ERROR ret = outputs[i].ToHost();
        if (ret != APP_ERR_OK) {
            LogError << "ToHost faile";
            return ret;
        }

        MxBase::MemoryData memorySrc = {};
        memorySrc.deviceId = outputs[i].GetDeviceId();
        memorySrc.type = (MxBase::MemoryData::MemoryType)outputs[i].GetTensorType();
        memorySrc.size = (uint32_t)outputs[i].GetSize();
        memorySrc.ptrData = outputs[i].GetBuffer();

        cv::Mat mat_result(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, CV_32FC3, memorySrc.ptrData);
        cv::cvtColor(mat_result, mat_result, cv::COLOR_RGB2BGR);
        mat_result = (mat_result + DE_MORMALIZE) * COVER_TO_COLOR;
        cv::resize(mat_result, mat_result, cv::Size(widthStride_, heightStride_));
        cv::Rect roi(0, 0, imageWidth_, imageHeight_);
        result = mat_result(roi);
    }
    return APP_ERR_OK;
}

APP_ERROR CartoonGANPicture::WriteResult(const cv::Mat &result, const std::string &imgPath)
{
    std::string fileName = imgPath.substr(imgPath.find_last_of("/") + 1);
    std::string directorPath = "./data/output/";

    size_t pre = 0;
    size_t pos = 0;
    std::string dir;
    int ret = 0;

    while((pos = directorPath.find_first_of('/', pre)) != std::string::npos){
        dir = directorPath.substr(0, pos++);
        pre = pos;
        if(dir.size() == 0) { continue; }
        ret = mkdir(dir.c_str(), S_IRWXU);
        if(ret && errno != EEXIST){
            LogError << "Directory is created failed, ret = " << ret << ".";
            return ret;
        }
    }
    cv::imwrite(directorPath + fileName, result);
    return APP_ERR_OK;
}

APP_ERROR CartoonGANPicture::Process(const std::string &imgPath)
{
    // process image
    MxBase::TensorBase inTensor;
    APP_ERROR ret = ReadImage(imgPath, inTensor);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    MxBase::TensorBase outTensor;
    ret = Resize(inTensor, outTensor);
    if (ret != APP_ERR_OK) {
        LogError << "Resize failed, ret=" << ret << ".";
        return ret;
    }
  
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    inputs.push_back(outTensor);
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    cv::Mat result;
    ret = PostProcess(outputs, result);
    if(ret != APP_ERR_OK){
        LogError << "PostProcess fail, ret = " << ret << ".";
        return ret;
    }

    ret = WriteResult(result, imgPath);
    if(ret != APP_ERR_OK){
        LogError << "WriteResult fail, ret = " << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}