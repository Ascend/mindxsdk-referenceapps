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
#include "Yolov5Detection.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include <unistd.h>
#include <sys/stat.h>
#include <string>

namespace{
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
    const uint32_t VPC_H_ALIGN = 2;
    const float CONFIDENCE = 0.25;
}

// 加载标签文件
APP_ERROR Yolov5Detection::LoadLabels(const std::string &labelPath, std::map<int, std::string> &labelMap) {
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
        if (s[0] == '#') {
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

// 设置配置参数
void Yolov5Detection::SetYolov5PostProcessConfig(const InitParam &initParam, 
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

APP_ERROR Yolov5Detection::Init(const InitParam &initParam) {
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
    SetYolov5PostProcessConfig(initParam, config);
    //  初始化Yolov5后处理对象
    post_ = std::make_shared<Yolov5PostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "Yolov5PostProcess init failed, ret=" << ret << ".";
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

APP_ERROR Yolov5Detection::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

// 获取图像数据，将数据存入TensorBase中
APP_ERROR Yolov5Detection::ReadImage(const std::string &imgPath, MxBase::TensorBase &tensor) {
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

APP_ERROR Yolov5Detection::Resize(const MxBase::TensorBase &inputTensor, MxBase::TensorBase &outputTensor) {
    auto shape = inputTensor.GetShape();
    MxBase::DvppDataInfo input = {};
    // 还原为原始尺寸
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

// 模型推理
APP_ERROR Yolov5Detection::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                           std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        // Yolov5模型3个检测特征图尺寸
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
       
        // 用3个检测特征图尺寸分别构建3个数据为空的tensor
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        // 将tensor存入outputs中
        outputs.push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    // 设置类型为静态batch
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

// 后处理
APP_ERROR Yolov5Detection::PostProcess(const MxBase::TensorBase &tensor, 
                                             const std::vector<MxBase::TensorBase> &outputs,
                                             std::vector<std::vector<MxBase::ObjectInfo>> &objInfos)
{
    // 通过原始图像tensor构建ResizedImageInfo
    auto shape = tensor.GetShape();
    MxBase::ResizedImageInfo imgInfo;
    imgInfo.widthOriginal = shape[1]; 
    imgInfo.heightOriginal = shape[0] * YUV_BYTE_DE / YUV_BYTE_NU; 

    imgInfo.widthResize = 416;
    imgInfo.heightResize = 416;
    imgInfo.resizeType = MxBase::RESIZER_STRETCHING;
    std::vector<MxBase::ResizedImageInfo> imageInfoVec = {};
    imageInfoVec.push_back(imgInfo);
    // use Yolov5PostProcess post_;
    APP_ERROR ret = post_->Process(outputs, objInfos, imageInfoVec);

    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }

    ret = post_->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "Yolov5PostProcess DeInit failed";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Yolov5Detection::WriteResult(MxBase::TensorBase &tensor,
                                            const std::vector<std::vector<MxBase::ObjectInfo>> &objInfos,
                                            const std::string &imgPath)
{
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << "ToHost faile";
        return ret;
    }
    auto shape = tensor.GetShape();
    // 初始化OpenCV图像信息矩阵
    cv::Mat imgYuv = cv::Mat(shape[0], shape[1], CV_8UC1, tensor.GetBuffer());
    cv::Mat imgBgr = cv::Mat(shape[0] * YUV_BYTE_DE / YUV_BYTE_NU, shape[1], CV_8UC3);
    // 颜色空间转换
    cv::cvtColor(imgYuv, imgBgr, cv::COLOR_YUV2BGR_NV12);

    uint32_t batchSize = objInfos.size();
    std::vector<MxBase::ObjectInfo> resultInfo;

    for (uint32_t i = 0; i < batchSize; i++) {
        for (uint32_t j = 0; j < objInfos[i].size(); j++) {
            // 将置信度大于阈值的结果存放进去
            if(objInfos[i][j].confidence > CONFIDENCE){
                resultInfo.push_back(objInfos[i][j]);
            }
        }
    }

    std::string newImgPath = imgPath.substr(imgPath.find_last_of("/") + 1);
    // 设置结果图片存放文件夹路径
    std::string newSavePath = "./result/" + newImgPath;
   
    for(uint32_t k = 0; k < resultInfo.size(); k++){
        // 打印置信度推理结果
        LogInfo << "id: " << resultInfo[k].classId << "; lable: " << resultInfo[k].className
                << "; confidence: " << resultInfo[k].confidence
                << "; box: [ (" << resultInfo[k].x0 << "," << resultInfo[k].y0 << ") "
                << "(" << resultInfo[k].x1-resultInfo[k].x0 << "," << resultInfo[k].y1 - resultInfo[k].y0 << ") ]" ;

        const cv::Scalar green = cv::Scalar(0, 255, 0);
        const uint32_t thickness = 4;
        const uint32_t xOffset = 10;
        const uint32_t yOffset = 10;
        const uint32_t lineType = 8;
        const float fontScale = 1.0;

        // 在图像上绘制文字
        cv::putText(imgBgr, resultInfo[k].className, cv::Point(resultInfo[k].x0 + xOffset, resultInfo[k].y0 + yOffset),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, green, thickness, lineType);
        // 绘制矩形
        cv::rectangle(imgBgr,cv::Rect(resultInfo[k].x0, resultInfo[k].y0,
                                      resultInfo[k].x1 - resultInfo[k].x0, resultInfo[k].y1 - resultInfo[k].y0),
                      green, thickness);
    }
    // 把Mat类型的图像矩阵保存为图像到指定位置。
    

    cv::imwrite(newSavePath, imgBgr);
    return APP_ERR_OK;
}

APP_ERROR Yolov5Detection::Process(const std::string &imgPath) {
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
    ret = WriteResult(inTensor, objInfos, imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "Save result failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}