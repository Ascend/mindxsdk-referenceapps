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
#ifndef SDKMEMORY_PLUGINOVERSPEED_H
#define SDKMEMORY_PLUGINOVERSPEED_H

#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/ErrorCode/ErrorCode.h"

/**
* @api
* @brief Definition of PluginOverSpeed class.
*/

namespace MxPlugins {
    class PluginOverSpeed : public MxTools::MxPluginBase {
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
        static MxTools::MxpiPortInfo DefineOutputPorts();

        /**
        * @api
        * @brief Define the output ports.
        * @return MxpiPortInfo
        */
        static int distance(int x0, int y0, int x1, int y1);

        /**
        * @api
        * @brief Calculate the distance between two points.
        * @param int
        * @return int
        */
        static int
        calculate(std::map<int, std::vector<int>> &trackdata, int confframes_, int &frame_num, int confthresh_,
                  std::shared_ptr<MxTools::MxpiTrackLetList> srcTrackLetListSptr,
                  std::shared_ptr<MxTools::MxpiObjectList> srcObjectListSptr);
        /**
        * @api
        * @brief Data processing.
        * @return int
        */
    private:
        APP_ERROR SetMxpiErrorInfo(MxTools::MxpiBuffer &buffer, const std::string pluginName,
                                   const MxTools::MxpiErrorInfo mxpiErrorInfo);

        std::string tracksource_;
        std::string detectionsource_;
        std::string descriptionMessage_;
        std::ostringstream ErrorInfo_;
        std::map<int, std::vector<int>> trackdata;
        int confthresh_;
        int confframes_;
        int confsleep_;
        int frame_num = 0;
        int sleeptime = 0;
        int frame = 0;
        int alarm_count = 0;
    };
}
#endif