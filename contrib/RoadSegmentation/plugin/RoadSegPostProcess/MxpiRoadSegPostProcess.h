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

#ifndef SDKMEMORY_MXPIROADSEGPOSTPROCESS_H
#define SDKMEMORY_MXPIROADSEGPOSTPROCESS_H
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
#include "opencv2/opencv.hpp"


/**
* @api
* @brief Definition of MxpiRoadSegPostProcesss class.
*/
namespace MxPlugins {
    struct ImageInfo {
            int modelWidth;
            int modelHeight;
            int imgWidth;
            int imgHeight;
    };
    class MxpiRoadSegPostProcess : public MxTools::MxPluginBase {
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
        
        /**
         * @api
         * @brief Get the number of class id and confidence from model inference.
         * @param key
         * @param buffer
         * @return APP_ERROR
         */
        APP_ERROR chw2hwc(const std::vector<MxBase::TensorBase> inputTensors,
                          std::vector<MxBase::TensorBase> &outputTensors);
        /**
         * @api
         * @brief convert NCHW to NHWC
         * @return APP_ERROR
         */
        APP_ERROR GenerateVisionList(const cv::Mat mask,const MxTools::MxpiVisionList srcMxpiVisionList,
                                     MxTools::MxpiVisionList& dstMxpiVisionList);
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
        
        APP_ERROR openCVImageFusion(size_t idx,const MxTools::MxpiVision srcMxpiVision,
                                    MxTools::MxpiVision& dstMxpiVision,
                                    cv::Mat threeChannelMask);
         /**
         * @api
         * @brief Fusion of original image and mask
         * @return APP_ERROR
         */
        APP_ERROR PostProcess(std::vector<MxBase::TensorBase> &inputTensors,
                              const ImageInfo &imageInfo, cv::Mat &mask);
         /**
         * @api
         * @brief   roadSegmentation PostProcess
         * @return  APP_ERROR
         */
        
        APP_ERROR GenerateVisionListOutput(const MxTools::MxpiTensorPackageList srcMxpiTensorPackageList,
                                           const MxTools::MxpiVisionList srcMxpiVisionList,
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
    };
}
#endif // SDKMEMORY_MXPIROADSEGPOSTPROCESS_H
