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
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "postprocess.h"


void (*my_fun)(const std::vector< std::vector<int64_t> >& seg_image,
    std::vector<READ_RESULT>* read_results,
    const int thread_num);

/**
* @api
* @brief Definition of MxpiSamplePlugin class.
*/
namespace MxPlugins {
    class Myplugin : public MxTools::MxPluginBase {
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
        static std::vector<std::shared_ptr<void>> DefineProperties();

    private:
        APP_ERROR SetMxpiErrorInfo(MxTools::MxpiBuffer& buffer, const std::string pluginName,
            const MxTools::MxpiErrorInfo mxpiErrorInfo);
        std::string parentName_;
        std::string descriptionMessage_;
        std::ostringstream ErrorInfo_;
    };
}

#endif // SDKMEMORY_MXPISAMPLEPLUGIN_H
