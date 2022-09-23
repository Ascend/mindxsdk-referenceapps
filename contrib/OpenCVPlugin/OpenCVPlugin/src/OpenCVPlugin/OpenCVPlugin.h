/*
* Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef SDKMEMORY_MXPISAMPLEPLUGIN_H
#define SDKMEMORY_MXPISAMPLEPLUGIN_H
#include "opencv2/opencv.hpp"
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"

/**
* @api
* @brief Definition of MxpiSamplePlugin class.
*/
namespace MxPlugins {
class MxpiSamplePlugin : public MxTools::MxPluginBase {
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
    APP_ERROR GenerateVisionList(const MxTools::MxpiVisionList srcMxpiVisionListptr,
                                 MxTools::MxpiVisionList& dstMxpiVisionListptr);
    APP_ERROR openCV(size_t idx, const MxTools::MxpiVision srcMxpiVision,
        		    MxTools::MxpiVision& dstMxpiVision);
    APP_ERROR Bgr2Yuv(cv::Mat src, cv::Mat &dst);
    APP_ERROR Mat2MxpiVisionOpencv(size_t idx, const cv::Mat& mat, MxTools::MxpiVision& vision);
    APP_ERROR Mat2MxpiVisionDvpp(size_t idx, const cv::Mat& mat, MxTools::MxpiVision& vision);
    void swapYUV_I420toNV12(unsigned char* i420bytes, unsigned char* nv12bytes, int width, int height);
    void Judge(auto& visionData, auto& visionInfo, cv::Mat &imgBgr, MxBase::MemoryData memorySrc,
                                      cv::Mat &src, MxBase::MemoryData memoryDst);
    void Output(cv::Mat dst, size_t idx, MxTools::MxpiVision& dstMxpiVision);
private:
    APP_ERROR SetMxpiErrorInfo(MxTools::MxpiBuffer& buffer, const std::string pluginName,
    const MxTools::MxpiErrorInfo mxpiErrorInfo);
    std::string parentName_;
    std::string descriptionMessage_;
    double startRow;
    double startCol;
    double endRow;
    double endCol;
    float height;
    float width;
    double fx;
    double fy;
    std::string outputDataFormat;
    std::string dataType;
    std::string option;
    int interpolation;
    std::ostringstream ErrorInfo_;
    MxBase::MxbasePixelFormat outputPixelFormat_;
    };
}
#endif
