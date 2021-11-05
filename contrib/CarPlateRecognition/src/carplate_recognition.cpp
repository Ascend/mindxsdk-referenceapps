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

#include "carplate_recognition.h"
#include "CvxText.h"
#include "opencv2/opencv.hpp"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

using namespace std;

// 以下3个参数用于YUV图像与BGR图像之间的宽高转换
namespace{
    const uint32_t YUV_BYTE_NU = 3; // 用于图像缩放
    const uint32_t YUV_BYTE_DE = 2; // 用于图像缩放
    const uint32_t VPC_H_ALIGN = 2; // 用于图像对齐
}


/* @brief: 初始化各类资源
   @param：initParam：初始化参数
   @retval:APP_ERROR型变量
*/
APP_ERROR CarPlateRecognition::init(const InitParam &initParam) {
   
    deviceId_ = initParam.deviceId; // 初始化设备ID
	
    // STEP1:资源初始化
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices(); 
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }	
    // STEP2:文本信息初始化
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }	
    // STEP3:初始化DvppWrapper,用于图片的编解码以及缩放
    dvppWrapper_ = std::make_shared<MxBase::DvppWrapper>();
    ret = dvppWrapper_->Init(); 
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper init failed, ret=" << ret << ".";
        return ret;
    }	
    // STEP4:载入车牌检测模型，将模型的描述信息分别写入变量detection_modelDesc_
    detection_model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = detection_model_->Init(initParam.DetecModelPath, detection_modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "Detection ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    // STEP5:载入车牌识别模型，将模型的描述信息分别写入变量recognition_modelDesc_
    recognition_model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = recognition_model_->Init(initParam.RecogModelPath, recognition_modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "Recognition ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    // STEP6:初始化车牌检测模型的后处理对象
    detection_post_ = std::make_shared<RetinaFacePostProcess>();
    ret = detection_post_->init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "RetinaFacePostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    // STEP7:初始化车牌识别模型的后处理对象
    recognition_post_ = std::make_shared<LPRPostProcess>();
    ret = recognition_post_->init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "LPRPostProcess init failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}


/* @brief: 释放各类资源
   @param：none
   @retval:APP_ERROR型变量
*/
APP_ERROR CarPlateRecognition::deinit() {
    dvppWrapper_->DeInit();
    detection_model_->DeInit();
    detection_post_->deinit();
    recognition_model_->DeInit();
    recognition_post_->deinit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}


/* @brief: 读取图片
   @param：imgPath：图片的存放路径
   @param：tensor：存储读取到的图片
   @retval:APP_ERROR型变量
*/
APP_ERROR CarPlateRecognition::readimage(const std::string &imgPath, MxBase::TensorBase &tensor) {

    // STEP1:对JPEG图像解码后存入DvppDataInfo中
    MxBase::DvppDataInfo output = {};   
    APP_ERROR ret = dvppWrapper_->DvppJpegDecode(imgPath, output);
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper DvppJpegDecode failed, ret=" << ret << ".";
        return ret;
    }
    // STEP2:将数据从HOST侧转移到DEVICE侧，以便后续处理
    MxBase::MemoryData memoryData((void*)output.data, output.dataSize,
                                    MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                                    deviceId_);
    // STEP3:对解码后图像对齐尺寸进行判定
    if (output.heightStride % VPC_H_ALIGN != 0) {
        LogError << "Output data height(" << output.heightStride << ") can't be divided by " << VPC_H_ALIGN << ".";
        MxBase::MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    // STEP4:将读取到的图片存入tensor
    std::vector<uint32_t> shape = {output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, output.widthStride};
    tensor = MxBase::TensorBase(memoryData, false, shape, MxBase::TENSOR_DTYPE_UINT8);

    return APP_ERR_OK;
}


/* @brief: 图像缩放，用在车牌检测推理前
   @param：inputTensor：原始的图像数据
   @param：outputTensor:缩放后的图像数据
   @retval:APP_ERROR型变量
*/
APP_ERROR CarPlateRecognition::resize(const MxBase::TensorBase &inputTensor, MxBase::TensorBase &outputTensor) {

    // STEP1:将图像还原为原始尺寸(在ReadImage的STEP4中图像进行了尺度变换)
    auto shape = inputTensor.GetShape();
    MxBase::DvppDataInfo input = {};    
    input.height = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.width = shape[1];
    input.heightStride = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.widthStride = shape[1];
    input.dataSize = inputTensor.GetByteSize();
    input.data = (uint8_t*)inputTensor.GetBuffer();
    // STEP2:进行图像缩放
    MxBase::ResizeConfig resize = {};
    resize.height = 640;
    resize.width = 640;
    MxBase::DvppDataInfo output = {};
    APP_ERROR ret = dvppWrapper_->VpcResize(input, output, resize);
    if (ret != APP_ERR_OK) {
        LogError << "VpcResize failed, ret=" << ret << ".";
        return ret;
    }
    // STEP3:将数据从HOST侧转移到DEVICE侧
    MxBase::MemoryData memoryData((void*)output.data, output.dataSize,
                                    MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
    // STEP4:对缩放后图像对齐尺寸进行判定
    if (output.heightStride % VPC_H_ALIGN != 0) {
        LogError << "Output data height(" << output.heightStride << ") can't be divided by " << VPC_H_ALIGN << ".";
        MxBase::MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    // STEP5:将缩放后的图片存入tensor
    shape = {output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, output.widthStride};
    outputTensor = MxBase::TensorBase(memoryData, false, shape,MxBase::TENSOR_DTYPE_UINT8);

    return APP_ERR_OK;
}


/* @brief: 图像缩放，被Crop_Resize1函数调用
   @param：inputTensor：原始的图像数据
   @param：outputTensor:缩放后的图像数据
   @retval:APP_ERROR型变量
*/
APP_ERROR CarPlateRecognition::resize1(const MxBase::TensorBase &inputTensor, MxBase::TensorBase &outputTensor) {
    // STEP1:将图像还原为原始尺寸(在ReadImage的STEP4中图像进行了尺度变换)
    auto shape = inputTensor.GetShape();
    MxBase::DvppDataInfo input = {};
    input.height = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.width = shape[1];
    input.heightStride = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.widthStride = shape[1];
    input.dataSize = inputTensor.GetByteSize();
    input.data = (uint8_t*)inputTensor.GetBuffer();
    // STEP2:进行图像缩放
    MxBase::ResizeConfig resize = {};
    resize.height = 72;
    resize.width = 272;
    MxBase::DvppDataInfo output = {};
    APP_ERROR ret = dvppWrapper_->VpcResize(input, output, resize);
    if (ret != APP_ERR_OK) {
        LogError << "VpcResize failed, ret=" << ret << ".";
        return ret;
    }
    // STEP3:将数据转为到DEVICE侧
    MxBase::MemoryData memoryData((void*)output.data, output.dataSize,
                                    MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
    // STEP4:对缩放后图像对齐尺寸进行判定
    if (output.heightStride % VPC_H_ALIGN != 0) {
        LogError << "Output data height(" << output.heightStride << ") can't be divided by " << VPC_H_ALIGN << ".";
        MxBase::MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    // STEP5:将缩放后的图片存入tensor
    shape = {output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, output.widthStride};
    outputTensor = MxBase::TensorBase(memoryData, false, shape,MxBase::TENSOR_DTYPE_UINT8);

    return APP_ERR_OK;
}


/* @brief: 抠图，被Crop_Resize1函数调用
   @param：inputTensor：原始的图像数据
   @param：outputTensor:抠图后的图像数据
   @param: objInfos:目标框信息
   @retval:APP_ERROR型变量
*/
APP_ERROR CarPlateRecognition::crop(const MxBase::TensorBase &inputTensor,
                                    MxBase::TensorBase &outputTensor,
                                    MxBase::ObjectInfo objInfo) {

    // STEP1:将图像还原为原始尺寸(在ReadImage的STEP4中图像进行了尺度变换)
    auto shape = inputTensor.GetShape();
    MxBase::DvppDataInfo input = {};
    input.height = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.width = shape[1];
    input.heightStride = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.widthStride = shape[1];
    input.dataSize = inputTensor.GetByteSize();
    input.data = (uint8_t*)inputTensor.GetBuffer();
    // STEP2:进行抠图
    MxBase::CropRoiConfig croproiconfig = {}; // 抠图配置参数
    croproiconfig.x0 = objInfo.x0;
    croproiconfig.x1 = objInfo.x1;
    croproiconfig.y0 = objInfo.y0;
    croproiconfig.y1 = objInfo.y1;
    MxBase::DvppDataInfo output = {};
    APP_ERROR ret = dvppWrapper_->VpcCrop(input, output, croproiconfig);
    if (ret != APP_ERR_OK) {
        LogError << "crop failed, ret=" << ret << ".";
        return ret;
    }
    // STEP3:将数据转为到DEVICE侧
    MxBase::MemoryData memoryData((void*)output.data, output.dataSize, 
                                MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                                deviceId_);
    // STEP4:对缩放后图像对齐尺寸进行判定
    if (output.heightStride % VPC_H_ALIGN != 0) {
        LogError << "Output data height(" << output.heightStride << ") can't be divided by " << VPC_H_ALIGN << ".";
        MxBase::MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    // STEP5:将缩放后的图片存入tensor
    shape = {output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, output.widthStride};
    outputTensor = MxBase::TensorBase(memoryData, false, shape,MxBase::TENSOR_DTYPE_UINT8);

    return APP_ERR_OK;
}


/* @brief: 抠图缩放函数，通过调用先前定义的Crop函数和Resize1函数实现，用在车牌识别推理前
   @param：inputTensor：原始的图像数据
   @param：outputTensors:抠图缩放后的图像数据
   @param：objInfos:目标框信息
   @retval:APP_ERROR型变量
*/
APP_ERROR CarPlateRecognition::crop_resize1(MxBase::TensorBase inputTensor,
                                            std::vector<MxBase::TensorBase> &cropresize_Tensors,
                                            std::vector<MxBase::ObjectInfo> objInfos)
{
    MxBase::TensorBase crop_Tensor;
    MxBase::TensorBase resize_Tensor;

    for(int i = 0; i<int(objInfos.size()); i++) // 遍历检测出来的所有目标框信息(因为在一幅图中可能检测出多个车牌)
    {
        APP_ERROR ret = crop(inputTensor, crop_Tensor, objInfos[i]);
        if (ret != APP_ERR_OK) {
            LogError << "crop failed, ret=" << ret << ".";
            return ret;
        }
        ret = resize1(crop_Tensor, resize_Tensor);
        if (ret != APP_ERR_OK) {
            LogError << "resize1 failed, ret=" << ret << ".";
            return ret;
        }
        cropresize_Tensors.push_back(resize_Tensor);
    }

    return APP_ERR_OK;
}


/* @brief: 进行车牌检测推理
   @param：inputs：输入数据
   @param：outputs:模型推理的输出数据
   @retval:APP_ERROR型变量
*/
APP_ERROR CarPlateRecognition::detection_inference(const std::vector<MxBase::TensorBase> inputs,
                                                    std::vector<MxBase::TensorBase> &outputs) {

    // STEP1:根据模型的输出创建空的TensorBase变量
    auto dtypes = detection_model_->GetOutputDataType();
    for (size_t i = 0; i < detection_modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < detection_modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)detection_modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    // STEP2:进行模型推理，结果存入outputs
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH; // 设置类型为静态batch
    APP_ERROR ret = detection_model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}


/* @brief: 车牌检测模型后处理
   @param：tensor：原始图像数据
   @param：outputs:模型的推理输出
   @param：objInfos:目标检测类任务的目标框信息
   @retval:APP_ERROR型变量
*/
APP_ERROR CarPlateRecognition::detection_postprocess(const MxBase::TensorBase orign_Tensor,
                                                    std::vector<MxBase::TensorBase> detect_outputs,
                                                    std::vector<MxBase::ObjectInfo>& objectInfos) {

    // STEP1:获取图像的缩放方式，用于后处理中的坐标还原使用
    auto shape = orign_Tensor.GetShape();
    MxBase::ResizedImageInfo resizedImageInfo;
    resizedImageInfo.widthOriginal = shape[1];
    resizedImageInfo.heightOriginal = shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    resizedImageInfo.widthResize = 640;
    resizedImageInfo.heightResize = 640;
    resizedImageInfo.resizeType = MxBase::RESIZER_STRETCHING;
    // STEP2:进行模型后处理
    APP_ERROR ret = detection_post_->process(detect_outputs, objectInfos, resizedImageInfo);
    if (ret != APP_ERR_OK) {
        LogError << "RetinaFacePostProcess failed, ret=" << ret << ".";
        return ret;
    }
    // STEP3:后处理完成，将资源释放
    ret = detection_post_->deinit();
    if (ret != APP_ERR_OK) {
        LogError << "RetinaFacePostProcess DeInit failed";
        return ret;
    }
    return APP_ERR_OK;
}


/* @brief: 进行车牌识别推理
*  @param：inputs：输入数据
*  @param：outputs:模型推理的输出数据
*  @retval:APP_ERROR型变量
*/
APP_ERROR CarPlateRecognition::recognition_inference(const std::vector<MxBase::TensorBase> inputs,
                                                    std::vector<std::vector<MxBase::TensorBase>> &outputs) {

    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH; // 设置类型为静态batch

    std::vector<MxBase::TensorBase> k_inputs;
    std::vector<MxBase::TensorBase> k_outputs;
    auto dtypes = recognition_model_->GetOutputDataType();
    for(int k = 0; k<int(inputs.size()); k++)
    {
        // STEP1:根据模型的输出创建空的TensorBase变量
        for (size_t i = 0; i < recognition_modelDesc_.outputTensors.size(); ++i) {
            std::vector<uint32_t> shape = {};
            for (size_t j = 0; j < recognition_modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
                shape.push_back((uint32_t)recognition_modelDesc_.outputTensors[i].tensorDims[j]);
            }
            MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
            APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
            if (ret != APP_ERR_OK) {
                LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
                return ret;
            }
            k_outputs.push_back(tensor);
        }
        // STEP2:进行车牌识别推理
        k_inputs.push_back(inputs[k]);
        APP_ERROR ret = recognition_model_->ModelInference(k_inputs, k_outputs, dynamicInfo);
        if (ret != APP_ERR_OK) {
            LogError << "ModelInference failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(k_outputs);
        k_inputs.clear();
        k_outputs.clear();
    }

    return APP_ERR_OK;
}


/* @brief: 车牌识别模型后处理
*  @param：tensor：原始图像数据
*  @param：outputs:模型的推理输出
*  @param：objInfos:其成员className用于存放所识别出来的车牌字符
*  @retval:APP_ERROR型变量
*/
APP_ERROR CarPlateRecognition::recognition_postprocess(std::vector<std::vector<MxBase::TensorBase>> recog_outputs,
                                                        std::vector<MxBase::ObjectInfo>& objectInfos) {

    for(int i = 0; i<int(objectInfos.size()); i++)
    {
         // STEP1:进行模型后处理
        APP_ERROR ret = recognition_post_->process(recog_outputs[i], objectInfos[i]);
        if (ret != APP_ERR_OK) {
            LogError << "LPRPostProcess failed, ret=" << ret << ".";
            return ret;
        }
	    // STEP2:后处理完成，将资源释放
        ret = recognition_post_->deinit();
        if (ret != APP_ERR_OK) {
            LogError << "LPRPostProcess DeInit failed";
            return ret;
        }
    }

    return APP_ERR_OK;
}


/* @brief: 车牌检测结果可视化
*  @param：tensor：未经Resize的原始图像数据
*  @param：objInfos：经模型后处理获得的目标框信息
*					因为可能检测到多个对象，每个对象又会产生多个目标框，所以objInfos是元素为ObjectInfo型容器的容器对象
*  @retval:APP_ERROR型变量
*/
APP_ERROR CarPlateRecognition::write_result(MxBase::TensorBase &tensor,
                                            std::vector<MxBase::ObjectInfo> objectInfos) {

    // STEP1:数据从DEVICE侧转到HOST侧（ReadImage的STEP2和Resize的STEP3将数据转到了DEVICE侧）
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << "ToHost faile";
        return ret;
    }
    // STEP2:初始化OpenCV图像信息矩阵,并进行颜色空间转换:YUV->BGR
    auto shape = tensor.GetShape();
    cv::Mat imgYuv = cv::Mat(shape[0], shape[1], CV_8UC1, tensor.GetBuffer());
    cv::Mat imgBgr = cv::Mat(shape[0] * YUV_BYTE_DE / YUV_BYTE_NU, shape[1], CV_8UC3);
    cv::cvtColor(imgYuv, imgBgr, cv::COLOR_YUV2BGR_NV12);
    // STEP3:对识别结果进行画框
    int baseline = 0; // 使用cv::getTextSize所必须传入的参数，但实际上未用到
    CvxText text("./simhei.ttf"); // 指定字体
    cv::Scalar size1{ 40, 0.5, 0.1, 0 }; // (字体大小, 无效的, 字符间距, 无效的 }
    text.set_font(nullptr, &size1, nullptr, 0);
    for (int j = 0; j < int(objectInfos.size()); ++j) {
        const char* str1 = objectInfos[j].className.data(); // 将std::string转为const char*
        char* str = (char*)(str1); // 将const char*转为char*（因为ToWchar只支持将char*转为wchar_t*）
        wchar_t *w_str = nullptr;
        text.to_wchar(str,w_str);

        // 写车牌号
        cv::Size text_size = cv::getTextSize(objectInfos[j].className, cv::FONT_HERSHEY_SIMPLEX, 1, 2, &baseline);
        text.put_text(imgBgr, w_str, cv::Point(objectInfos[j].x0 + (objectInfos[j].x1 - objectInfos[j].x0) / 2 
                        - text_size.width / 2 + 5,objectInfos[j].y0 - 5), cv::Scalar(0, 0, 255));
        // 画目标框
        cv::Rect rect(objectInfos[j].x0, objectInfos[j].y0, objectInfos[j].x1 - objectInfos[j].x0,
                        objectInfos[j].y1 - objectInfos[j].y0);
        cv::rectangle(imgBgr, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
    }
	// STEP4:将结果保存为图片
    cv::imwrite("./result.jpg", imgBgr);

    return APP_ERR_OK;
}


/* @brief: 串联整体流程
*  @param：imgPath：图片路径
*  @retval:APP_ERROR型变量
*/
APP_ERROR CarPlateRecognition::process(const std::string &imgPath) {

	// STEP1:读取图片
    MxBase::TensorBase orign_Tensor;
    APP_ERROR ret = readimage(imgPath, orign_Tensor);
    if (ret != APP_ERR_OK) {
        LogError << "readimage failed, ret=" << ret << ".";
        return ret;
    }

	// STEP2:图片缩放为640*640
    MxBase::TensorBase resize_Tensor;
    ret = resize(orign_Tensor, resize_Tensor);
    if (ret != APP_ERR_OK) {
        LogError << "resize failed, ret=" << ret << ".";
        return ret;
    }

	// STEP3:车牌检测模型推理
    std::vector<MxBase::TensorBase> detect_inputs = {};
    std::vector<MxBase::TensorBase> detect_outputs = {};
    detect_inputs.push_back(resize_Tensor);
    ret = detection_inference(detect_inputs, detect_outputs);
    if (ret != APP_ERR_OK) {
        LogError << "detection_inference failed, ret=" << ret << ".";
        return ret;
    }

	// STEP4:车牌检测模型后处理
    std::vector<MxBase::ObjectInfo> objInfos;
    ret = detection_postprocess(orign_Tensor, detect_outputs, objInfos);
    if (ret != APP_ERR_OK) {
        LogError << "detection_postprocess failed, ret=" << ret << ".";
        return ret;
    }

	// STEP5:将检测到的车牌抠图，并缩放至72*272
    std::vector<MxBase::TensorBase> cropresize_Tensors = {};
    ret = crop_resize1(orign_Tensor, cropresize_Tensors, objInfos);
    if (ret != APP_ERR_OK) {
        LogError << "resize1 failed, ret=" << ret << ".";
        return ret;
    }

	// STEP6:车牌识别模型推理
    std::vector<MxBase::TensorBase> recog_inputs = {};
    std::vector<std::vector<MxBase::TensorBase>> recog_outputs = {};
    recog_inputs = cropresize_Tensors;
    ret = recognition_inference(recog_inputs, recog_outputs);
    if (ret != APP_ERR_OK) {
        LogError << "recognition_inference failed, ret=" << ret << ".";
        return ret;
    }

	// STEP7:车牌识别模型后处理
    ret = recognition_postprocess(recog_outputs, objInfos);
    if (ret != APP_ERR_OK) {
        LogError << "recognition_postprocess failed, ret=" << ret << ".";
        return ret;
    }

	// STEP8:结果可视化
    write_result(orign_Tensor, objInfos);

    return APP_ERR_OK;
}

