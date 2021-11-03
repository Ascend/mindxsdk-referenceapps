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

#ifndef CarPlate_Recognition_H
#define CarPlate_Recognition_H

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "MxBase/ConfigUtil/ConfigUtil.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "retinaface_postprocess.h"
#include "lpr_postprocess.h"
#include "initparam.h"


// 车牌识别大类，其中调用了车牌检测模型后处理类RetinaFace_PostProcess和车牌识别模型后处理类LPR_PostProcess
class CarPlate_Recognition{
public:
    APP_ERROR Init(const InitParam &initParam); // 初始化函数
    APP_ERROR DeInit(); // 解初始化函数
    APP_ERROR Detection_Inference(const std::vector<MxBase::TensorBase> inputs, std::vector<MxBase::TensorBase> &outputs); // 车牌检测推理函数
    APP_ERROR Detection_PostProcess(const MxBase::TensorBase orign_Tensor, std::vector<MxBase::TensorBase> detect_outputs, std::vector<MxBase::ObjectInfo>& objectInfos); // 车牌检测后处理函数
	APP_ERROR Recognition_Inference(const std::vector<MxBase::TensorBase> inputs, std::vector<std::vector<MxBase::TensorBase>> &outputs); // 车牌识别推理函数
    APP_ERROR Recognition_PostProcess(std::vector<std::vector<MxBase::TensorBase>> recog_outputs, std::vector<MxBase::ObjectInfo>& objectInfos); // 车牌识别后处理函数
    APP_ERROR Process(const std::string &imgPath); // 整体流程处理函数
protected:
    APP_ERROR ReadImage(const std::string &imgPath, MxBase::TensorBase &tensor); // 读取图像函数
    APP_ERROR Resize(const MxBase::TensorBase &inputTensor, MxBase::TensorBase &outputTensor);  // 图像缩放函数
    APP_ERROR Resize1(const MxBase::TensorBase &inputTensor, MxBase::TensorBase &outputTensor); // 图像缩放函数，被Crop_Resize1函数调用
	APP_ERROR Crop(const MxBase::TensorBase &inputTensor, MxBase::TensorBase &outputTensor, MxBase::ObjectInfo objInfo); // 抠图函数，被Crop_Resize1函数调用
	APP_ERROR Crop_Resize1(MxBase::TensorBase inputTensor, std::vector<MxBase::TensorBase> &cropresize_Tensors, std::vector<MxBase::ObjectInfo> objInfos); // 抠图缩放函数
    APP_ERROR WriteResult(MxBase::TensorBase &tensor, std::vector<MxBase::ObjectInfo> objectInfos); // 结果可视化函数
private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_; // DvppWrapper对象，封装了图像解码、缩放、扣图等功能
    std::shared_ptr<MxBase::ModelInferenceProcessor> detection_model_;   // 车牌检测模型对象
	std::shared_ptr<MxBase::ModelInferenceProcessor> recognition_model_; // 车牌识别模型对象
    std::shared_ptr<RetinaFace_PostProcess> detection_post_; // 车牌检测模型后处理对象
	std::shared_ptr<LPR_PostProcess> recognition_post_;	 // 车牌识别模型后处理对象
    MxBase::ModelDesc detection_modelDesc_   = {}; // 车牌检测模型描述信息
	MxBase::ModelDesc recognition_modelDesc_ = {}; // 车牌识别模型描述信息
    uint32_t deviceId_ = 0; // 设备ID
};

#endif
