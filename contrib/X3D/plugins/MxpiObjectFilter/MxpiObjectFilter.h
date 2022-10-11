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
#ifndef SDKMEMORY_MXPIOBJECTFILTER_H
#define SDKMEMORY_MXPIOBJECTFILTER_H

#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxTools/PluginToolkit/buffer/MxpiBufferManager.h"
#include "MxBase/Log/Log.h"

/**
* @api
* @brief Definition of MxpiObjectFilter class.
*/
namespace MxPlugins {
class MxpiObjectFilter : public MxTools::MxPluginBase {
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
private:
    /**
     * @api
     * @brief Check metadata.
     * @param MxTools::MxpiMetadataManager.
     * @return Error Code.
     */
    APP_ERROR CheckDataSource(MxTools::MxpiMetadataManager &mxpiMetadataManager);
    std::string dataSource_;
    MxTools::InputParam inputParam;
    std::ostringstream ErrorInfo_;
};
}
#endif //SDKMEMORY_MXPIOBJECTFILTER_H
