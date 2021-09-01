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
#include <RcfPostProcess.h>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    bool checkTensor;
    std::string modelPath;
    uint32_t outSizeNum;
    std::string outSize;
    uint32_t rcfType;
    uint32_t modelType;
    uint32_t inputType;
};

class RcfDetection {
public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const MxBase::TensorBase &tensor, const std::vector<MxBase::TensorBase> &outputs,
                          std::vector<MxBase::TensorBase> &postProcessOutput);
    APP_ERROR Process(const std::string &imgPath);
protected:
    APP_ERROR ReadImage(const std::string &imgPath, MxBase::TensorBase &tensor);
    APP_ERROR Resize(const MxBase::TensorBase &inputTensor, MxBase::TensorBase &outputTensor,uint32_t resizeHeight,uint32_t resizeWidth);
    APP_ERROR WriteResult(MxBase::TensorBase &inferTensor, const std::string &imgPath);
    void SetRcfPostProcessConfig(const InitParam &initParam, std::map<std::string, std::shared_ptr<void>> &config);

private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    std::shared_ptr<RcfPostProcess> post_;
    MxBase::ModelDesc modelDesc_ = {};
    uint32_t deviceId_ = 0;
    int dvppHeightStride;
    int dvppWidthStride;
};
#endif
