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

#include "CrowdCount.h"
#include "MxBase/Log/Log.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp> 
#include <opencv2/highgui.hpp> 
#include "MxBase/Maths/FastMath.h"

namespace {
    auto uint8Deleter = [](uint8_t *p) {};
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
    const uint32_t VPC_H_ALIGN = 2;

}
using namespace MxBase;
using namespace cv;
APP_ERROR CrowdCountPostProcess::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) {
    LogDebug << "Start to Init CrowdCountPostProcess.";
    APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
        return ret;
    }
    configData_.GetFileValue<int>("MODEL_WIDTH", modelWidth_);
    configData_.GetFileValue<int>("MODEL_HEIGHT", modelHeight_);
    LogDebug << "End to Init CrowdCountPostProcess.";
    return APP_ERR_OK;
}

APP_ERROR CrowdCountPostProcess::DeInit() {
    LogInfo << "Begin to deinitialize CrowdCountPostProcessor.";
	LogInfo << "End to deinitialize CrowdCountPostProcessor.";
	return APP_ERR_OK;
}

void CrowdCountPostProcess::ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
					         std::vector<MxBase::TensorBase> &outputs,
					         const std::vector<ResizedImageInfo> &resizedImageInfos) {
    LogDebug << "CrowdCountPostProcess start to write results.";
    if (tensors.size() == 0) {
        return;
    }
    auto shape = tensors[0].GetShape();
    if (shape.size() == 0) {
        return;
    }
    uint32_t batchSize = shape[0];
    for (uint32_t i = 0; i < batchSize; i++) { 
	std::vector<std::shared_ptr<void>> featLayerData = {};
        std::vector<std::vector<size_t>> featLayerShapes = {};
        for (uint32_t j = 0; j < tensors.size(); j++) {
            auto dataPtr = (uint8_t *) GetBuffer(tensors[j], i);
            std::shared_ptr<void> tmpPointer;
            tmpPointer.reset(dataPtr, uint8Deleter);
            featLayerData.push_back(tmpPointer);
            shape = tensors[j].GetShape();
            std::vector<size_t> featLayerShape = {};
            for (auto s : shape) {
                featLayerShape.push_back((size_t) s);
            }
            featLayerShapes.push_back(featLayerShape);
        }
        GenerateHeatmap(tensors, outputs, resizedImageInfos[i].widthResize,
                     resizedImageInfos[i].heightResize);
    }
    LogDebug << "CrowdCountPostProcess write results successed.";
}

APP_ERROR CrowdCountPostProcess::Process(const std::vector<TensorBase> &tensors,
					 std::vector<MxBase::TensorBase> &outputs,
				         const std::vector<ResizedImageInfo> &resizedImageInfos,
                                         const std::map<std::string, std::shared_ptr<void>> &paramMap) {
    LogDebug << "Start to Process CrowdCountPostProcess.";
    APP_ERROR ret = APP_ERR_OK;
    if (resizedImageInfos.size() == 0) {
        ret = APP_ERR_INPUT_NOT_MATCH;
        LogError << GetError(ret) << "resizedImageInfos is not provided which is necessary for CrowdCountPostProcess.";
        return ret;
    }
    auto inputs = tensors;
    ret = CheckAndMoveTensors(inputs);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "CheckAndMoveTensors failed.";
        return ret;
    }
    ObjectDetectionOutput(inputs, outputs ,resizedImageInfos);
    LogDebug << "End to Process CrowdCountPostProcess.";
    return APP_ERR_OK;
}

void CrowdCountPostProcess::GenerateHeatmap(const std::vector <TensorBase> &tensors, 
                                            std::vector<MxBase::TensorBase> &outputs,
                                            const int netWidth, const int netHeight) {									 
     NetInfo netInfo;   
     netInfo.netWidth = netWidth;
     netInfo.netHeight = netHeight;
     cv::Mat heatMap;
     cv::Mat colorMat;
     auto data = tensors[0];
     auto shape = data.GetShape();
     cv::Mat modelOutput = cv::Mat(shape[2], shape[3], CV_32FC1, data.GetBuffer());
     modelOutput.convertTo(heatMap, CV_8UC1, 255.0/255);
     GaussianBlur(heatMap, heatMap, Size(0, 0), 5.0, 5.0,BORDER_DEFAULT);
     normalize(heatMap, heatMap, 0, 255, NORM_MINMAX, CV_8UC1);
     applyColorMap(heatMap, colorMat, COLORMAP_JET);
     heatMap = colorMat;
     void *ptr = tensors[0].GetBuffer();
     float sum = std::accumulate((float*)ptr , (float*)ptr + tensors[0].GetSize(), 0.f);
     sum /= 1000.f;
     LogInfo << "person count sum:" << sum;
     heatMap = colorMat;
     shape = {1, shape[2], shape[3], 3};
     MxBase::TensorBase output(shape, MxBase::TensorDataType::TENSOR_DTYPE_UINT8, MxBase::MemoryData::MemoryType::MEMORY_HOST_NEW, -1);
     APP_ERROR  ret = MxBase::TensorBase::TensorBaseMalloc(output);
     MxBase::MemoryData srcMemory(heatMap.data, heatMap.rows * heatMap.cols, MxBase::MemoryData::MemoryType::MEMORY_HOST_NEW, 0);
     MxBase::TensorBase heatMapTensor(srcMemory, true, shape, MxBase::TensorDataType::TENSOR_DTYPE_UINT8);
     std::copy((uint8_t*)heatMap.data, (uint8_t*)heatMap.data + heatMapTensor.GetByteSize(), (uint8_t*)output.GetBuffer());
     outputs.push_back(output);
     LogDebug << "End to Process CrowdCountPostProcess.";
}



