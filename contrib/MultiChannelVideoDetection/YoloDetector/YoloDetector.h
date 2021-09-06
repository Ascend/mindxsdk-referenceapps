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

#ifndef MULTICHANNELVIDEODETECTION_YOLODETECTOR_H
#define MULTICHANNELVIDEODETECTION_YOLODETECTOR_H

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "ObjectPostProcessors/Yolov3PostProcess.h"

namespace AscendYoloDetector {
// yolo config
const uint32_t YOLO_MODEL_INPUT_WIDTH = 416;
const uint32_t YOLO_MODEL_INPUT_HEIGHT = 416;

const uint32_t YOLO_CLASS_NUM = 80;
const uint32_t YOLO_BIASES_NUM = 18;
const uint32_t YOLO_TYPE = 3;
const uint32_t YOLO_MODEL_TYPE = 0;
const uint32_t YOLO_INPUT_TYPE = 0;
const uint32_t YOLO_ANCHOR_DIM = 3;

// yuv parameter
const uint32_t YUV_BYTE_NU = 3;
const uint32_t YUV_BYTE_DE = 2;
const uint32_t VPC_H_ALIGN = 2;

struct YoloInitParam {
    uint32_t deviceId = 0;
    std::string labelPath;
    bool checkTensor = true;
    std::string modelPath;
    uint32_t classNum = YOLO_CLASS_NUM;
    uint32_t biasesNum = YOLO_BIASES_NUM;
    std::string biases;
    std::string objectnessThresh;
    std::string iouThresh;
    std::string scoreThresh;
    uint32_t yoloType = YOLO_TYPE;
    uint32_t modelType = YOLO_MODEL_TYPE;
    uint32_t inputType = YOLO_INPUT_TYPE;
    uint32_t anchorDim = YOLO_ANCHOR_DIM;
};

struct PostProcessConfig {
    uint32_t originWidth = 0;
    uint32_t originHeight = 0;
    uint32_t modelWidth = 0;
    uint32_t modelHeight = 0;
};

class YoloDetector {
public:
    YoloDetector() = default;
    ~YoloDetector() = default;

    APP_ERROR Init(const YoloInitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Detect(const MxBase::DvppDataInfo &imageInfo,
                     const PostProcessConfig &postProcessConfig,
                     std::vector<std::vector<MxBase::ObjectInfo>> &objInfos);
    APP_ERROR Inference(const MxBase::DvppDataInfo &imageInfo, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &modelOutputs,
                          const PostProcessConfig &postProcessConfig,
                          std::vector<std::vector<MxBase::ObjectInfo>> &objInfos);

protected:
    static APP_ERROR LoadLabels(const std::string &labelPath, std::map<int, std::string> &labelMap);
    static APP_ERROR LoadPostProcessConfig(const YoloInitParam &initParam,
                                           std::map<std::string, std::shared_ptr<void>> &config);

private:
    APP_ERROR InitModel(const YoloInitParam &initParam);
    APP_ERROR InitPostProcess(const YoloInitParam &initParam);
    APP_ERROR TransformImageToTensor(const MxBase::DvppDataInfo &imageInfo, MxBase::TensorBase &tensor) const;
    APP_ERROR InternalInference(const std::vector<MxBase::TensorBase> &inputs,
                                std::vector<MxBase::TensorBase> &outputs);

private:
    // model
    std::shared_ptr<MxBase::ModelInferenceProcessor> model;
    // infer result post process
    std::shared_ptr<MxBase::Yolov3PostProcess> postProcess;
    // yolo model desc which contain input and output message
    MxBase::ModelDesc modelDesc = {};
    // yolo label map
    std::map<int, std::string> labelMap = {};

    // device id
    uint32_t deviceId;
};
} // end AscendYoloDetector
#endif // MULTICHANNELVIDEODETECTION_YOLODETECTOR_H