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

#include "RcfPostProcess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"

namespace {
    const float weight1 =  0.2009036;
    const float weight2 =  0.2101715;
    const float weight3 =  0.22262956;
    const float weight4 =  0.22857015;
    const float weight5 =  0.2479302;
    const float weight6 =  0.00299916;
    
    auto uint8Deleter = [](uint8_t *p) {};
}
using namespace MxBase;

APP_ERROR RcfPostProcess::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) {
    LogDebug << "Start to Init RcfPostProcess.";
    LogDebug << "End to Init RcfPostProcess.";
    return APP_ERR_OK;
}

APP_ERROR RcfPostProcess::DeInit() {
    return APP_ERR_OK;
}


static APP_ERROR ResizeTensor(const MxBase::TensorBase &input, MxBase::TensorBase &output, const uint32_t &width, const uint32_t &height)
{
    
    auto inputShape = input.GetShape();
    uint32_t h = inputShape[2];
    uint32_t w = inputShape[3];
    cv::Mat inputMat = cv::Mat(h, w, CV_32FC1, input.GetBuffer());
    cv::Mat outputMat;
    cv::resize(inputMat, outputMat, cv::Size(width, height));
    std::vector<uint32_t> outputShape = {1, 1, height, width};
    MxBase::TensorBase tensor(outputShape, MxBase::TensorDataType::TENSOR_DTYPE_FLOAT32, MxBase::MemoryData::MemoryType::MEMORY_HOST_NEW, -1);
    APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "TensorBaseMalloc failed.";
        return ret;
    }
    std::copy((uint8_t*)outputMat.data, (uint8_t*)outputMat.data + tensor.GetByteSize(), (uint8_t*)tensor.GetBuffer());
    output = tensor;
    return APP_ERR_OK;
}

APP_ERROR RcfPostProcess::Process(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs)
{
    
    auto tensors = inputs;
    APP_ERROR ret = CheckAndMoveTensors(tensors);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "CheckAndMoveTensors failed.";
        return ret;
    }

    const uint32_t resizeWidth = 512;
    const uint32_t resizeHeight = 512;
    std::vector<MxBase::TensorBase> resizeTensors = {};
    for (auto input : inputs) {
        MxBase::TensorBase output;
        ret = ResizeTensor(input, output, resizeWidth, resizeHeight);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "ResizeTensor failed.";
            return ret;
        }
        resizeTensors.push_back(output);
    }

    std::vector<uint32_t> outputShape = {1, 1, resizeHeight, resizeWidth};
    MxBase::TensorBase output(outputShape, MxBase::TensorDataType::TENSOR_DTYPE_FLOAT32, MxBase::MemoryData::MemoryType::MEMORY_HOST_NEW, -1);
    ret = MxBase::TensorBase::TensorBaseMalloc(output);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "TensorBaseMalloc failed.";
        return ret;
    }

    if (resizeTensors.size() != 5) {
        LogError << "resizeTensors.size():" << resizeTensors.size();
        return APP_ERR_COMM_FAILURE;
    }

    const uint32_t firstLayerIndex = 0;
    const uint32_t secondLayerIndex = 1;
    const uint32_t thirdLayerIndex = 2;
    const uint32_t fouthLayerIndex = 3;
    const uint32_t fifthtLayerIndex = 4;
    const uint32_t scale = 255;
    auto ptr1 = (float*)resizeTensors[firstLayerIndex].GetBuffer();
    auto ptr2 = (float*)resizeTensors[secondLayerIndex].GetBuffer();
    auto ptr3 = (float*)resizeTensors[thirdLayerIndex].GetBuffer();
    auto ptr4 = (float*)resizeTensors[fouthLayerIndex].GetBuffer();
    auto ptr5 = (float*)resizeTensors[fifthtLayerIndex].GetBuffer();

    auto dst = (float*)output.GetBuffer();

    for (uint32_t i = 0; i < output.GetSize(); i++) {
        float value = weight1 * ptr1[i] + weight2 * ptr2[i] + weight3 * ptr3[i] + weight4 * ptr4[i] + weight5 * ptr5[i] + weight6;
        dst[i] = fastmath::sigmoid(value) * scale;
    }

    outputs.push_back(output);
    

    return APP_ERR_OK;
}
