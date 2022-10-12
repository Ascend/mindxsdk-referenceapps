/*
 * Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

#ifndef SDKMEMORY_MXPIPNETPOSTPROCESS_H
#define SDKMEMORY_MXPIPNETPOSTPROCESS_H

#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "opencv2/opencv.hpp"
#include <cstdint>

#if defined(__cpp_lib_math_special_functions) || !defined(DNUMCPP_NO_USE_BOOST)
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Functions.hpp"
#endif
#include <type_traits>

#ifndef __cpp_lib_math_special_functions
#include <cmath>
#else
#include "boost/math/special_functions/bessel.hpp"
#endif

/**
* @api
* @brief Definition of MxpiHeadPosePlugin class.
*/

namespace MxPlugins {
    class MxpiPNetPostprocess : public MxTools::MxPluginBase {
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
        APP_ERROR GenerateHeadPoseInfo(const MxTools::MxpiTensorPackageList srcMxpiTensorPackage,
                                       const MxTools::MxpiVisionList srcMxpiVisionPackage,
                                       MxTools::MxpiObjectList& dstMxpiHeadPoseListSptr);

        nc::NdArray<float> transform_preds(nc::NdArray<float> coords, nc::NdArray<float> center,
                                           float scale, int output_size[2]);

        cv::Mat get_affine_transform(nc::NdArray<float> center, float scale, float rot, int output_size[2],
                                     nc::NdArray<float> shift, int inv);

        nc::NdArray<float> get_dir(nc::NdArray<float> src_point, float rot_rad);

        nc::NdArray<float> get_3rd_point(nc::NdArray<float> a, nc::NdArray<float> b);

        nc::NdArray<float> affine_transform(nc::NdArray<float> pt, nc::NdArray<float> t);

    private:
        APP_ERROR SetMxpiErrorInfo(MxTools::MxpiBuffer& buffer, const std::string pluginName,
                                   const MxTools::MxpiErrorInfo mxpiErrorInfo);
        std::string parentName_;
        std::string infoPlugName_;
        std::string descriptionMessage_;
        std::ostringstream ErrorInfo_;
    };
}


#endif // SDKMEMORY_MXPIPNETPOSTPROCESS_H
