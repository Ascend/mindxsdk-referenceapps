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

#ifndef MXBASE_CROUNDCOUNT_H
#define MXBASE_CROUNDCOUNT_H

#include <opencv2/opencv.hpp>
#include <CrowdCountPostProcess.h>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"


extern std::vector<double> g_inferCost;
struct InitParam {
    uint32_t deviceId;
    bool checkTensor;
    std::string modelPath;
    uint32_t classNum;
    std::string labelPath;
};
class CrowdCount{
public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &images);
    APP_ERROR Process(const std::string &imgPath);
protected:
    APP_ERROR ReadImage(const std::string &imgPath, MxBase::TensorBase &tensor);
    APP_ERROR Resize(const MxBase::TensorBase &inputTensor, MxBase::TensorBase &outputTensor);
    APP_ERROR WriteResult(const std::vector<MxBase::TensorBase> &outputs,  const std::vector<MxBase::TensorBase> &postimageMat);			 
    void SetCrowdCountPostProcessConfig(const InitParam &initParam,
                                                       std::map<std::string, std::shared_ptr<void>> &config);
private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    std::shared_ptr<CrowdCountPostProcess> post_;
    MxBase::ModelDesc modelDesc_ = {};
    uint32_t deviceId_ = 0;
    uint32_t imageWidth_ = 0;
    uint32_t imageHeight_ = 0;

};
#endif
