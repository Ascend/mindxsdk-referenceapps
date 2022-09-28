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
#ifndef SDKMEMORY_MXPIX3DPREPROCESS_H
#define SDKMEMORY_MXPIX3DPREPROCESS_H

#include "opencv2/opencv.hpp"
#include "MxBase/Log/Log.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
#include "MxStream/StreamManager/MxStreamManager.h"
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/buffer/MxpiBufferManager.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxTools/Proto/MxpiDataTypeDeleter.h"
#include "MxTools/Proto/MxpiDataType.pb.h"

/**
* @api
* @brief Definition of MxpiX3DPreProcess class.
*/
namespace MxPlugins {
class MxpiX3DPreProcess : public MxTools::MxPluginBase {
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

    std::shared_ptr<MxTools::MxpiTensorPackageList> StackFrame(uint32_t start_idx);

    static std::vector<std::shared_ptr<void>> DefineProperties();
    /**
     * @api
     * @brief Get the number of class id and confidence from model inference.
     * @param key
     * @param buffer
     * @return APP_ERROR
     */
private:
    /**
     * @api
     * @brief Check metadata.
     * @param MxTools::MxpiMetadataManager.
     * @return Error Code.
     */
    APP_ERROR CheckDataSource(MxTools::MxpiMetadataManager &mxpiMetadataManager);
    APP_ERROR PreProcessVision(MxTools::MxpiVision srcMxpiVision, uint32_t insert_idx);
    std::string savePath_;
    std::string dataSource_;
    uint32_t skipFrameNum_ = 0;
    uint32_t windowStride_ = 0;
    MxTools::InputParam inputParam;
    std::ostringstream ErrorInfo_;
};
}
#endif //SDKMEMORY_MXPIX3DPREPROCESS_H
