/*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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

#ifndef MXPLUGINS_MXPISKIPFRAME_H
#define MXPLUGINS_MXPISKIPFRAME_H

#include "./MindX_SDK/mxVision/include/MxBase/ErrorCode/ErrorCode.h"
#include "./MindX_SDK/mxVision/include/MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "./MindX_SDK/mxVision/include/MxTools/PluginToolkit/buffer/MxpiBufferManager.h"
#include "./MindX_SDK/mxVision/include/MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "./MindX_SDK/mxVision/include/MxTools/Proto/MxpiDataType.pb.h"

/**
 * This plugin is used for skip frame.
 */
namespace MxPlugins {
    class MxpiSelectedFrame : public MxTools::MxPluginBase {
    public:
        /**
        * @description: Init configs.
        * @param configParamMap: config.
        * @return: Error code.
        */
        APP_ERROR Init(std::map<std::string, std::shared_ptr<void>> &configParamMap) override;

        /**
        * @description: DeInit device.
        * @return: Error code.
        */
        APP_ERROR DeInit() override;

        /**
        * @description: MxpiSelectedFrame plugin process.
        * @param mxpiBuffer: data receive from the previous.
        * @return: Error code.
        */
        APP_ERROR Process(std::vector<MxTools::MxpiBuffer*> &mxpiBuffer) override;

        /**
        * @description: MxpiSelectedFrame plugin define properties.
        * @return: properties.
        */
        static std::vector<std::shared_ptr<void>> DefineProperties();

        /**
        * @api
        * @brief Define the number and data type of input ports.
        * @return MxTools::MxpiPortInfo.
        */
        static MxTools::MxpiPortInfo DefineInputPorts();

        /**
        * @api
        * @brief Define the number and data type of output ports.
        * @return MxTools::MxpiPortInfo.
        */
        static MxTools::MxpiPortInfo DefineOutputPorts();

    private:

        uint32_t SelectedFrameNum_ = 0;

        uint32_t count = 0;
    };
}

#endif