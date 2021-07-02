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

#ifndef SDKMEMORY_PLUGINVIOLENTACTTON_H
#define SDKMEMORY_PLUGINVIOLENTACTTON_H

#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"

/**
 * This plug is to recognize whether the object's action is a Violent Action and alarm.
*/

namespace MxPlugins {
    class PluginViolentAction : public MxTools::MxPluginBase {
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
        * @description: Plugin_ViolentAction plugin process.
        * @param mxpiBuffer: data receive from the previous.
        * @return: Error code.
        */
        APP_ERROR Process(std::vector<MxTools::MxpiBuffer *> &mxpiBuffer) override;

        /**
        * @description: Plugin_ViolentAction plugin define properties.
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
        /**
         * @api
         * @brief Check metadata.
         * @param MxTools::MxpiMetadataManager.
         * @return Error Code.
         */
        APP_ERROR CheckDataSource(MxTools::MxpiMetadataManager &mxpiMetadataManager);

        /**
        * @api
        * @brief Match the recognized action class to the action of interest and alarm.
        * @return MxTools::MxpiAttributeList.
        */
        std::shared_ptr<MxTools::MxpiAttributeList> ActionMatch(std::shared_ptr<MxTools::MxpiClassList> &mxpiClassList);

        void ReadTxt(std::string file, std::vector<std::string> &aoi);  // read the action of interest txt file
        std::string classSource_ = "";    // previous plugin MxpiClassList
        std::string filePath_ = "";
        std::ostringstream ErrorInfo_;     // Error Code
        std::vector<std::string> aoi = {};
        std::string alarmInformation = "";
        int pathflag = 0;                 // flag to mark file IO only once
        uint32_t sleepTime_ = 0;
        uint32_t detectSleep_ = 0;
        int alarm_count = 0;
        float actionThreshold_ = 0.0;
    };
}
#endif