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
#pragma once
#ifndef SDKMEMORY_MXPIPREPROCESS_H
#define SDKMEMORY_MXPIPREPROCESS_H
#include <algorithm>
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"

#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
#include "opencv2/opencv.hpp"


/**
* @api
* @brief Definition of MxpiPostProcess class.
*/
namespace mx_plugins {
    struct ImageInfo {
            int modelWidth;
            int modelHeight;
            int imgWidth;
            int imgHeight;
    };
    class MxpiPostProcess : public MxTools::MxPluginBase {
    public:
       
        /**
         * @api
         * @brief Initialize configure parameter.
         * @param configParamMap
         * @return APP_ERROR
         */
        APP_ERROR Init(std::map<std::string, std::shared_ptr<void>>& configParamMap) override;
        /**
         * @api
         * @brief DeInitialize configure parameter.
         * @return APP_ERROR
         */
        APP_ERROR DeInit() override;
        /**
         * @api
         * @brief Process the data of MxpiBuffer.
         * @param mxpiBuffer
         * @return APP_ERROR
         */
        APP_ERROR Process(std::vector<MxTools::MxpiBuffer*>& mxpiBuffer) override;
        
        /**
         * @api
         * @brief Definition the parameter of configure properties.
         * @return std::vector<std::shared_ptr<void>>
         */
        static std::vector<std::shared_ptr<void>> DefineProperties();
        
        APP_ERROR GenerateVisionList(const cv::Mat mask, MxTools::MxpiVisionList& dstMxpiVisionList);
         /**
         * @api
         * @brief   get  original image and fuse with mask
         * @return  APP_ERROR
         */
        APP_ERROR Mat2MxpiVision(size_t idx, const cv::Mat& mat, MxTools::MxpiVision& vision);
         /**
         * @api
         * @brief mat to MxpiVisionList
         * @return APP_ERROR
         */
        
        APP_ERROR PostProcess(std::vector<MxBase::TensorBase> &inputTensors,
                              uint32_t imgHeight, uint32_t imgWidth, cv::Mat &mask);
         /**
         * @api
         * @brief   roadSegmentation PostProcess
         * @return  APP_ERROR
         */
        
        APP_ERROR GenerateVisionListOutput(const MxTools::MxpiTensorPackageList srcMxpiTensorPackageList,
                                           MxTools::MxpiVisionList& dstMxpiVisionList);
         /**
         * @api
         * @brief   get infered tensor and decoded image output result
         * @return  APP_ERROR
         */
    private:
        APP_ERROR SetMxpiErrorInfo(MxTools::MxpiBuffer& buffer, const std::string pluginName,
                                   const MxTools::MxpiErrorInfo mxpiErrorInfo);
        std::string parentName_;
        std::ostringstream ErrorInfo_;
        int index;
    };
}
#endif // SDKMEMORY_MXPIPREPROCESS_H
