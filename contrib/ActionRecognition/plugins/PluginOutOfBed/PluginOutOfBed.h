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

#ifndef MXPIFAIRMOT_FAIRMOT_OUTOFBED_H
#define MXPIFAIRMOT_FAIRMOT_OUTOFBED_H

#include <map>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"

/**
* @api
* @brief Definition of PluginOutOfBed class.
*/

namespace MxPlugins {
    class PluginOutOfBed : public MxTools::MxPluginBase {
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
            * @brief read ROI
            * @param ROI file path
            */
        void readTxt(std::string file, std::vector<cv::Point> &roi);

        /**
           * @api
           * @brief Out of bed detection.
           * @param queue
           * @return bool
           */
        bool OutOfBed(std::vector<int> queue);

        /**
           * @api
           * @brief Out of bed process.
           * @param srcTrackLetListSptr ,srcTrackLetListSptr
           * @return bool
           */
        bool OutOfBedProcess(std::shared_ptr<MxTools::MxpiTrackLetList> srcTrackLetListSptr,
                             std::shared_ptr<MxTools::MxpiObjectList> srcObjectListSptr);

    private:
        APP_ERROR SetMxpiErrorInfo(MxTools::MxpiBuffer &buffer, const std::string pluginName,
                                   const MxTools::MxpiErrorInfo mxpiErrorInfo);

        std::string tracksource_;
        std::string detectionsource_;
        std::string descriptionMessage_;
        std::string configpath;
        std::ostringstream ErrorInfo_;
        uint confthres_;
        float confratio_;
        uint confsleep_;
        uint sleeptime = 0;
        std::map<uint32_t, std::vector<int>> trackdata;
        std::vector<cv::Point> bed;
        uint frames;
        uint framesnum = 0;
        bool pathflag = true;
        uint alarmcount = 0;
    };

}
#endif // MXPIFAIRMOT_FAIRMOT_OUTOFBED_H
