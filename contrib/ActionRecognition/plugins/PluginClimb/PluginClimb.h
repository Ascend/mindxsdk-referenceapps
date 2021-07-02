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
#ifndef SDKMEMORY_PLUGINCLIMB_H
#define SDKMEMORY_PLUGINCLIMB_H

#include <opencv2/opencv.hpp>
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"

using namespace std;

/**
* @api
* @brief Definition of PluginClimb class.
*/

namespace MxPlugins {
    class PluginClimb : public MxTools::MxPluginBase {
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

        static MxTools::MxpiPortInfo DefineOutputPorts();

        static void readTxt(std::string file, std::vector<cv::Point> &roi);

        /**
        * @api
        * @brief log in the txt of roi.
        * @param std::string file
        * @return td::vector <cv::Point> &roi
        */
        static int distance(int x0, int y0, int x1, int y1);

        /**
        * @api
        * @brief Calculate the distance between two points.
        * @param int
        * @return int
        */
        int calculate(int bufferlength_, int highthresh_, float ratio_, vector<cv::Point> roi,
                      std::shared_ptr<MxTools::MxpiTrackLetList> srcTrackLetListSptr,
                      std::shared_ptr<MxTools::MxpiObjectList> srcObjectListSptr);
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
        std::map<int, std::vector<int>> trackdata;
        int confthresh_;
        int confframes_;
        int confsleep_;
        int confhigh_;
        int frame_num = 0;
        int sleeptime = 0;
        int pathflag = 0;

        int test_framenum = 0;
        int highthresh_;
        int bufferlength_;
        float ratio_;
        int detectsleep_;
        int alarm_count = 0;
        std::string filepath_;
        std::vector<cv::Point> roi;
    };
}
#endif