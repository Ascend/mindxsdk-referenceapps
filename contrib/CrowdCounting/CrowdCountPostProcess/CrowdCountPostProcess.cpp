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
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
    const uint32_t VPC_H_ALIGN = 2;

}
using namespace MxBase;
using namespace cv;
APP_ERROR CrowdCountPostProcess::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) {
    LogDebug << "Start to Init CrowdCountPostProcess.";
    APP_ERROR ret = PostProcessBase::Init(postConfig);
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

APP_ERROR CrowdCountPostProcess::Process(const std::vector <TensorBase> &tensors, 
                                            std::vector<MxBase::TensorBase> &outputs) {									 
    LogDebug << "Start to Process CrowdCountPostProcess.";
    auto inputs = tensors;
    APP_ERROR ret = APP_ERR_OK;
    ret = CheckAndMoveTensors(inputs);
    if (ret != APP_ERR_OK) {
       LogError << GetError(ret) << "CheckAndMoveTensors failed.";
       return ret;
    }
    cv::Mat heatMap;
    cv::Mat colorMat;
    float scale = 255.0/255;
    double sigmaX = 5.0;
    double sigmaY = 5.0;
    double alpha = 0;
    double beta = 255;
    float division_num = 1000.0;
    auto data = tensors[0];
    auto shape = data.GetShape();
    int batch_size = 1;
    int channels = 3;
    auto height = shape[2];
    auto width = shape[3];
    cv::Mat modelOutput = cv::Mat(height, width, CV_32FC1, data.GetBuffer());
    modelOutput.convertTo(heatMap, CV_8UC1, scale);
    GaussianBlur(heatMap, heatMap, Size(0, 0), sigmaX, sigmaY,BORDER_DEFAULT);
    normalize(heatMap, heatMap, alpha, beta, NORM_MINMAX, CV_8UC1);
    applyColorMap(heatMap, colorMat, COLORMAP_JET);
    heatMap = colorMat;
    void *ptr = tensors[0].GetBuffer();
    float sum = std::accumulate((float*)ptr , (float*)ptr + tensors[0].GetSize(), 0.f);
    sum /= division_num;
    int result = round(sum);
    LogInfo << "person count sum:" << result;
    heatMap = colorMat;
    // heatMap的shape(H,W,C) 
    shape = {batch_size, height, width, channels};
    // cv::Mat转tensor
    MxBase::TensorBase output(shape, MxBase::TensorDataType::TENSOR_DTYPE_UINT8, MxBase::MemoryData::MemoryType::MEMORY_HOST_NEW, -1);
    // 给tensor申请内存空间
    ret = MxBase::TensorBase::TensorBaseMalloc(output);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "TensorBaseMalloc failed.";
        return ret;
    }
    MxBase::MemoryData srcMemory(heatMap.data, heatMap.rows * heatMap.cols, MxBase::MemoryData::MemoryType::MEMORY_HOST_NEW, 0);
    MxBase::TensorBase heatMapTensor(srcMemory, true, shape, MxBase::TensorDataType::TENSOR_DTYPE_UINT8);
    std::copy((uint8_t*)heatMap.data, (uint8_t*)heatMap.data + heatMapTensor.GetByteSize(), (uint8_t*)output.GetBuffer());
    outputs.push_back(output);
    LogDebug << "End to Process CrowdCountPostProcess.";
}



