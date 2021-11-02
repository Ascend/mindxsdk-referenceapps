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

#ifndef MXBASE_YOLOV3DETECTIONOPENCV_H
#define MXBASE_YOLOV3DETECTIONOPENCV_H

#include <opencv2/opencv.hpp>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
struct InitParam {
    uint32_t deviceId;
    std::string modelPath;
};

class CartoonGANPicture {
public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Process(const std::string &imgPath);

private:
    APP_ERROR ReadImage(const std::string &imgPath, MxBase::TensorBase &tensor);
    APP_ERROR Resize(const MxBase::TensorBase &inputTensor, MxBase::TensorBase &outputTensor);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(std::vector<MxBase::TensorBase> &outputs, cv::Mat &result);
    APP_ERROR WriteResult(const cv::Mat &result, const std::string &imgPath);

private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_ = {};
    uint32_t deviceId_ = 0;
    uint32_t imageWidth_ = 0;
    uint32_t imageHeight_ = 0;
    uint32_t widthStride_ = 0;
    uint32_t heightStride_ = 0;
};
#endif