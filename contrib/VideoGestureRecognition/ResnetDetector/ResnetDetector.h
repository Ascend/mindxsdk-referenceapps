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


#ifndef MULTICHANNELVIDEODETECTION_ResnetDETECTOR_H
#define MULTICHANNELVIDEODETECTION_ResnetDETECTOR_H

#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "ClassPostProcessors/Resnet50PostProcess.h"

#include "../BlockingQueue/BlockingQueue.h"

namespace AscendResnetDetector {

struct ResnetInitParam {
    uint32_t deviceId;
    std::string labelPath;
    bool checkTensor;
    std::string modelPath;
    uint32_t classNum;
    uint32_t biasesNum;
    std::string biases;
    std::string objectnessThresh;
    std::string iouThresh;
    std::string scoreThresh;
    uint32_t resnetType;
    uint32_t modelType;
    uint32_t inputType;
    uint32_t anchorDim;
};

class ResnetDetector {
public:
    ResnetDetector() = default;
    ~ResnetDetector() = default;

    APP_ERROR Init(const ResnetInitParam & initParam);
    APP_ERROR DeInit();
    APP_ERROR Process();

    APP_ERROR Detect(const MxBase::DvppDataInfo &imageInfo, std::vector<std::vector<MxBase::ClassInfo>> &objInfos,
                     const uint32_t &imageOriginWidth, const uint32_t &imageOriginHeight);

private:
    APP_ERROR InitModel(const ResnetInitParam &initParam);
    APP_ERROR InitPostProcess(const ResnetInitParam &initParam);
    APP_ERROR TransformImageToTensor(const MxBase::DvppDataInfo &imageInfo, MxBase::TensorBase &tensor) const;
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &modelOutputs,const uint32_t &width,
                          const uint32_t &height, std::vector<std::vector<MxBase::ClassInfo>> &objInfos);

protected:
    static APP_ERROR LoadLabels(const std::string &labelPath, std::map<int, std::string> &labelMap);
    static APP_ERROR LoadPostProcessConfig(const ResnetInitParam &initParam, std::map<std::string, std::shared_ptr<void>> &config);

public:
    // running flag
    bool stopFlag;

private:
    // model
    std::shared_ptr<MxBase::ModelInferenceProcessor> model;
    // infer result post process
    std::shared_ptr<MxBase::Resnet50PostProcess> postProcess;
    MxBase::ModelDesc modelDesc = {};
    std::map<int, std::string> labelMap = {};

    // device id
    uint32_t deviceId;
};
} // end AscendResnetDetector
#endif //MULTICHANNELVIDEODETECTION_ResnetDETECTOR_H
