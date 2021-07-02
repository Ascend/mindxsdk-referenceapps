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

#ifndef SDKMEMORY_PLUGINALONE_H
#define SDKMEMORY_PLUGINALONE_H

#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/ErrorCode/ErrorCode.h"

using namespace MxTools;
/**
* @api
* @brief Definition of PluginAlone class.
*/

namespace MxPlugins {
    class PluginAlone : public MxTools::MxPluginBase {
    public:
        /**
        * @api
        * @brief Initialize configure parameter.
        * @param configParamMap
        * @return APP_ERROR
        */
        APP_ERROR Init(std::map<std::string, std::shared_ptr<void>> &configParamMap) override;

        /**
        * * @api
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
        APP_ERROR Process(std::vector<MxTools::MxpiBuffer *> &mxpiBuffer) override;

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
        static MxTools::MxpiPortInfo DefineInputPorts();

        /**
        * @api
        * @brief Define the input ports.
        * @return MxpiPortInfo
        */
        static int calculate(std::vector<int> &data_queue, int confthres_, float confratio_,
                             std::shared_ptr<MxpiTrackLetList> srcTrackLetListSptr,
                             std::shared_ptr<MxpiObjectList> srcObjectListSptr);
        /**
        * @api
        * @brief Data processing.
        * @return int
        */
    private:
        APP_ERROR SetMxpiErrorInfo(MxTools::MxpiBuffer &buffer,
                                   const std::string pluginName,
                                   const MxTools::MxpiErrorInfo mxpiErrorInfo);

        std::string tracksource_;
        std::string detectionsource_;
        std::string descriptionMessage_;
        std::ostringstream ErrorInfo_;
        int confthres_;
        float confratio_;
        int confsleep_;
        int sleeptime = 0;
        std::vector<int> data_queue;
        int alarm_count = 0;
    };
}
#endif // SDKMEMORY_PLUGINALONE_H