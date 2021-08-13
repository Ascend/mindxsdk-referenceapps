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

#include "YoloDetector.h"

#include "MxBase/Tensor/TensorContext/TensorContext.h"

namespace AscendYoloDetector {

APP_ERROR YoloDetector::Init(const YoloInitParam &initParam)
{
    LogInfo << "YoloDetector init start.";
    this->deviceId = initParam.deviceId;

    // set tensor context
    APP_ERROR ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set tensor context failed, ret = " << ret << ".";
        return ret;
    }

    // Init yolo model
    ret = InitModel(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "init model failed.";
        return ret;
    }

    // Init model post process
    ret = InitPostProcess(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "init model post process failed.";
        return ret;
    }

    // Load labels map
    ret = LoadLabels(initParam.labelPath, labelMap);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to load labels, ret = " << ret << ".";
        return ret;
    }

    LogInfo << "YoloDetector init successful.";
    return APP_ERR_OK;
}

APP_ERROR YoloDetector::DeInit()
{
    LogInfo << "YoloDetector deinit start.";

    APP_ERROR ret = model->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "deinit model failed.";
        return ret;
    }

    ret = postProcess->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "deinit model postprocess failed.";
        return ret;
    }

    LogInfo << "YoloDetector deinit successful.";
    return APP_ERR_OK;
}

APP_ERROR YoloDetector::Process()
{
    return APP_ERR_OK;
}

APP_ERROR YoloDetector::Detect(const MxBase::DvppDataInfo &imageInfo,
                               std::vector<std::vector<MxBase::ObjectInfo>> &objInfos,
                               const uint32_t &imageOriginWidth, const uint32_t &imageOriginHeight,
                               const uint32_t &modelWidth, const uint32_t &modelHeight)
{
    // transform image data to tensor
    MxBase::TensorBase tensor;
    APP_ERROR ret = TransformImageToTensor(imageInfo, tensor);
    if (ret != APP_ERR_OK) {
        LogError << "Transform image to tensor failed, ret = " << ret << ".";
        return ret;
    }

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    inputs.push_back(tensor);

    // model infer
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Model infer failed, ret = " << ret << ".";
        return ret;
    }

    // model post process
    ret = PostProcess(outputs, imageOriginWidth, imageOriginHeight, modelWidth, modelHeight, objInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Model post process failed, ret = " << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

/// ========== private Method ========== ///

APP_ERROR YoloDetector::InitModel(const YoloInitParam &initParam)
{
    LogInfo << "YoloDetector init model start.";
    model = std::make_shared<MxBase::ModelInferenceProcessor>();
    LogInfo << "model path: " << initParam.modelPath;

    APP_ERROR ret = model->Init(initParam.modelPath, modelDesc);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    LogInfo << "YoloDetector init model successfully.";
    return APP_ERR_OK;
}

APP_ERROR YoloDetector::InitPostProcess(const YoloInitParam &initParam)
{
    LogInfo << "YoloDetector init postprocess start.";
    std::map<std::string, std::shared_ptr<void>> config;
    LoadPostProcessConfig(initParam, config);

    postProcess = std::make_shared<MxBase::Yolov3PostProcess>();
    APP_ERROR ret = postProcess->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "ModelPostProcess init failed, ret=" << ret << ".";
        return ret;
    }

    LogInfo << "YoloDetector init postprocess successfully.";
    return APP_ERR_OK;
}

APP_ERROR YoloDetector::TransformImageToTensor(const MxBase::DvppDataInfo &imageInfo, MxBase::TensorBase &tensor) const
{
    MxBase::MemoryData memoryData((void*) imageInfo.data, imageInfo.dataSize,
                                  MxBase::MemoryData::MEMORY_DVPP, deviceId);
    if (imageInfo.heightStride % VPC_H_ALIGN != 0) {
        LogError << "Output data height(" << imageInfo.heightStride << ") can't be divided by " << VPC_H_ALIGN << ".";
        MxBase::MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }

    std::vector<uint32_t> shape = {imageInfo.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, imageInfo.widthStride};
    tensor = MxBase::TensorBase(memoryData, false, shape, MxBase::TENSOR_DTYPE_UINT8);

    return APP_ERR_OK;
}

APP_ERROR YoloDetector::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                  std::vector<MxBase::TensorBase> &outputs)
{
    APP_ERROR ret;

    // check input
    if (inputs.empty() || inputs[0].GetBuffer() == nullptr) {
        LogError << "input is null.";
        return APP_ERR_FAILURE;
    }

    auto dtypes = model->GetOutputDataType();
    for (size_t i = 0; i < modelDesc.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (long tensorDim : modelDesc.outputTensors[i].tensorDims) {
            shape.push_back((uint32_t) tensorDim);
        }

        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DVPP, (int32_t) deviceId);
        ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret = " << ret << ".";
            return ret;
        }

        outputs.push_back(tensor);
    }

    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;

    // model infer
    ret = model->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret = " << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR YoloDetector::PostProcess(const std::vector<MxBase::TensorBase> &modelOutputs,
                                    const uint32_t &originWidth, const uint32_t &originHeight,
                                    const uint32_t &modelWidth, const uint32_t &modelHeight,
                                    std::vector<std::vector<MxBase::ObjectInfo>> &objInfos)
{
    MxBase::ResizedImageInfo imageInfo = {};
    imageInfo.widthOriginal = originWidth;
    imageInfo.heightOriginal = originHeight;
    imageInfo.widthResize = modelWidth;
    imageInfo.heightResize = modelHeight;
    imageInfo.resizeType = MxBase::RESIZER_STRETCHING;
    std::vector<MxBase::ResizedImageInfo> imageInfoVec = {};
    imageInfoVec.push_back(imageInfo);

    APP_ERROR ret = postProcess->Process(modelOutputs, objInfos, imageInfoVec);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret = " << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR YoloDetector::LoadLabels(const std::string &labelPath, std::map<int, std::string> &labelMap)
{
    LogInfo << "load model labels start.";

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

    LogInfo << "load model labels successfully.";
    return APP_ERR_OK;
}

APP_ERROR YoloDetector::LoadPostProcessConfig(const YoloInitParam &initParam,
                                              std::map<std::string, std::shared_ptr<void>> &config)
{
    LogInfo << "load postprocess config start.";

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

    LogInfo << "load postprocess config successfully.";
    return APP_ERR_OK;
}

} // end AscendYoloDetector
