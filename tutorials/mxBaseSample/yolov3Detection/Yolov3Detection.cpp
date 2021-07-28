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
#include "Yolov3Detection.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include <unistd.h>
#include <sys/stat.h>

namespace{
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
    const uint32_t VPC_H_ALIGN = 2;
}

APP_ERROR Yolov3DetectionOpencv::LoadLabels(const std::string &labelPath, std::map<int, std::string> &labelMap) {
    std::ifstream infile;
    // open label file
    infile.open(labelPath, std::ios_base::in);
    std::string s;
    // check label file validity
    if (infile.fail()) {
        LogError << "Failed to open label file: " << labelPath << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    labelMap.clear();
    // construct label map
    int count = 0;
    while (std::getline(infile, s)) {
        if (s.find('#') <= 1) {
            continue;
        }
        size_t eraseIndex = s.find_last_not_of("\r\n\t");
        if (eraseIndex != std::string::npos) {
            s.erase(eraseIndex + 1, s.size() - eraseIndex);
        }
        labelMap.insert(std::pair<int, std::string>(count, s));
        count++;
    }
    infile.close();
    return APP_ERR_OK;
}

void Yolov3DetectionOpencv::SetYolov3PostProcessConfig(const InitParam &initParam,
                                                       std::map<std::string, std::shared_ptr<void>> &config) {
    MxBase::ConfigData configData;
    const std::string checkTensor = initParam.checkTensor ? "true" : "false";
    configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
    configData.SetJsonValue("BIASES_NUM", std::to_string(initParam.biasesNum));
    configData.SetJsonValue("BIASES", initParam.biases);
    configData.SetJsonValue("OBJECTNESS_THRESH", initParam.objectnessThresh);
    configData.SetJsonValue("IOU_THRESH", initParam.iouThresh);
    configData.SetJsonValue("SCORE_THRESH", initParam.scoreThresh);
    configData.SetJsonValue("YOLO_TYPE", std::to_string(initParam.yoloType));
    configData.SetJsonValue("MODEL_TYPE", std::to_string(initParam.modelType));
    configData.SetJsonValue("INPUT_TYPE", std::to_string(initParam.inputType));
    configData.SetJsonValue("ANCHOR_DIM", std::to_string(initParam.anchorDim));
    configData.SetJsonValue("CHECK_MODEL", checkTensor);

    auto jsonStr = configData.GetCfgJson().serialize();
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);
}

APP_ERROR Yolov3DetectionOpencv::Init(const InitParam &initParam) {
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
    SetYolov3PostProcessConfig(initParam, config);
    post_ = std::make_shared<Yolov3PostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "Yolov3PostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    // load labels from file
    ret = LoadLabels(initParam.labelPath, labelMap_);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to load labels, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Yolov3DetectionOpencv::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR Yolov3DetectionOpencv::ReadImage(const std::string &imgPath, MxBase::TensorBase &tensor) {
    MxBase::DvppDataInfo output = {};
    APP_ERROR ret = dvppWrapper_->DvppJpegDecode(imgPath, output);
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper DvppJpegDecode failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::MemoryData memoryData((void*)output.data, output.dataSize, MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
    if (output.heightStride % VPC_H_ALIGN != 0) {
        LogError << "Output data height(" << output.heightStride << ") can't be divided by " << VPC_H_ALIGN << ".";
        MxBase::MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    std::vector<uint32_t> shape = {output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, output.widthStride};
    tensor = MxBase::TensorBase(memoryData, false, shape, MxBase::TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR Yolov3DetectionOpencv::Resize(const MxBase::TensorBase &inputTensor, MxBase::TensorBase &outputTensor) {
    auto shape = inputTensor.GetShape();
    MxBase::DvppDataInfo input = {};
    input.height = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.width = shape[1];
    input.heightStride = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.widthStride = shape[1];
    input.dataSize = inputTensor.GetByteSize();
    input.data = (uint8_t*)inputTensor.GetBuffer();
    const uint32_t resizeHeight = 416;
    const uint32_t resizeWidth = 416;
    MxBase::ResizeConfig resize = {};
    resize.height = resizeHeight;
    resize.width = resizeWidth;
    MxBase::DvppDataInfo output = {};
    APP_ERROR ret = dvppWrapper_->VpcResize(input, output, resize);
    if (ret != APP_ERR_OK) {
        LogError << "VpcResize failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::MemoryData memoryData((void*)output.data, output.dataSize, MxBase::MemoryData::MemoryType::MEMORY_DVPP, deviceId_);
    if (output.heightStride % VPC_H_ALIGN != 0) {
        LogError << "Output data height(" << output.heightStride << ") can't be divided by " << VPC_H_ALIGN << ".";
        MxBase::MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    shape = {output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, output.widthStride};
    outputTensor = MxBase::TensorBase(memoryData, false, shape,MxBase:: TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR Yolov3DetectionOpencv::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                           std::vector<MxBase::TensorBase> &outputs) {
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

APP_ERROR Yolov3DetectionOpencv::PostProcess(const MxBase::TensorBase &tensor, const std::vector<MxBase::TensorBase> &outputs,
                                             std::vector<std::vector<MxBase::ObjectInfo>> &objInfos)
{
    auto shape = tensor.GetShape();
    MxBase::ResizedImageInfo imgInfo;
    imgInfo.widthOriginal = shape[1];
    imgInfo.heightOriginal = shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    imgInfo.widthResize = 416;
    imgInfo.heightResize = 416;
    imgInfo.resizeType = MxBase::RESIZER_STRETCHING;
    std::vector<MxBase::ResizedImageInfo> imageInfoVec = {};
    imageInfoVec.push_back(imgInfo);
    // use Yolov3PostProcess post_;
    APP_ERROR ret = post_->Process(outputs, objInfos, imageInfoVec);

    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }

    ret = post_->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "Yolov3PostProcess DeInit failed";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Yolov3DetectionOpencv::WriteResult(MxBase::TensorBase &tensor,
                                            const std::vector<std::vector<MxBase::ObjectInfo>> &objInfos)
{
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << "ToHost faile";
        return ret;
    }
    auto shape = tensor.GetShape();
    // 用输出原件信息初始化OpenCV图像信息矩阵
    cv::Mat imgYuv = cv::Mat(shape[0], shape[1], CV_8UC1, tensor.GetBuffer());
    cv::Mat imgBgr = cv::Mat(shape[0] * YUV_BYTE_DE / YUV_BYTE_NU, shape[1], CV_8UC3);
    // 颜色空间转换
    cv::cvtColor(imgYuv, imgBgr, cv::COLOR_YUV2BGR_NV12);

    uint32_t batchSize = objInfos.size();
    std::vector<MxBase::ObjectInfo> resultInfo;
    for (uint32_t i = 0; i < batchSize; i++) {
        float maxConfidence = 0;
        uint32_t  index;
        for (uint32_t j = 0; j < objInfos[i].size(); j++) {
            if(objInfos[i][j].confidence > maxConfidence) {
                maxConfidence = objInfos[i][j].confidence;
                index = j;
            }
        }
        resultInfo.push_back(objInfos[i][index]);
        LogInfo << "id: " << resultInfo[i].classId << "; lable: " << resultInfo[i].className
                << "; confidence: " << resultInfo[i].confidence
                << "; box: [ (" << resultInfo[i].x0 << "," << resultInfo[i].y0 << ") "
                << "(" << resultInfo[i].x1 << "," << resultInfo[i].y1 << ") ]" ;

        const cv::Scalar green = cv::Scalar(0, 255, 0);
        const uint32_t thickness = 4;
        const uint32_t xOffset = 10;
        const uint32_t yOffset = 10;
        const uint32_t lineType = 8;
        const float fontScale = 1.0;

        // 在图像上绘制文字
        cv::putText(imgBgr, resultInfo[i].className, cv::Point(resultInfo[i].x0 + xOffset, resultInfo[i].y0 + yOffset),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, green, thickness, lineType);
        // 绘制矩形
        cv::rectangle(imgBgr,cv::Rect(resultInfo[i].x0, resultInfo[i].y0,
                                      resultInfo[i].x1 - resultInfo[i].x0, resultInfo[i].y1 - resultInfo[i].y0),
                      green, thickness);
    }
    // 把Mat类型的图像矩阵保存为图像到指定位置。
    cv::imwrite("./result.jpg", imgBgr);
    return APP_ERR_OK;
}

APP_ERROR Yolov3DetectionOpencv::Process(const std::string &imgPath) {
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

    std::vector<std::vector<MxBase::ObjectInfo>> objInfos;
    ret = PostProcess(inTensor, outputs, objInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    ret = WriteResult(inTensor, objInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Save result failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}