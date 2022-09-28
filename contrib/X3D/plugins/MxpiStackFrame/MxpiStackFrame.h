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
 * imitations under the License.
 */

#ifndef MXPLUGINS_MXPIFRAMESTACK_H
#define MXPLUGINS_MXPIFRAMESTACK_H

#include <chrono>
#include <string>
#include <map>
#include "opencv2/opencv.hpp"
#include "MxBase/Log/Log.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
#include "MxStream/StreamManager/MxStreamManager.h"
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/buffer/MxpiBufferManager.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxTools/Proto/MxpiDataTypeDeleter.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "BlockingMap.h"

/**
 * This plugin is to stack frames based on detected objects.
*/

namespace MxPlugins {
    class MxpiStackFrame : public MxTools::MxPluginBase {
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
        * @description: MxpiStackFrame plugin process.
        * @param mxpiBuffer: data receive from the previous.
        * @return: Error code.
        */
        APP_ERROR Process(std::vector<MxTools::MxpiBuffer *> &mxpiBuffer) override;

        /**
        * @description: MxpiStackFrame plugin define properties.
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

        /**
         * @api
         * @brief Transfer MxpiVisionList to MxpiTensorPackageList.
         * @param shared_ptr <MxpiVisionList>.
         * @return shared_ptr <MxpiTensorPackageList>.
         */
        static std::shared_ptr<MxTools::MxpiTensorPackageList>
        ConvertVisionList2TensorPackageList(std::shared_ptr<MxTools::MxpiVisionList> &mxpiVisionList);

        /**
         * @api
         * @brief Reconstruct MxpiVisionList according to MxpiVisionData.
         * @param std::vector<MxTools::MxpiVisionData>.
         * @return shared_ptr <MxTools::MxpiVisionList>.
         */
        static std::shared_ptr<MxTools::MxpiVisionList>
        ConstructMxpiVisionList(std::vector<MxTools::MxpiVisionData> &slidingWindow);

        /**
         * @api
         * @brief Convert MxpiVisionData to MemoryData.
         * @param MxTools::MxpiVisionData.
         * @return MxBase::MemoryData.
         */
        static MxBase::MemoryData ConvertMemoryData(const MxTools::MxpiVisionData &mxpiVisionData);

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
         * @brief Stack frames for each human object.
         * @param MxpiVisionList, MxpiTrackLetList, BlockingMap
         */
        void StackFrame(const std::shared_ptr<MxTools::MxpiVisionList> &visionList,
                        const std::shared_ptr<MxTools::MxpiTrackLetList> &trackLetList,
                        std::shared_ptr<BlockingMap> &blockingMap);

        // Check Thread function; Check whether a object is 8 frames
        void CheckFrames(std::shared_ptr<BlockingMap> &blockingMap);

        void CreateThread();       // create CheckFrame thread
        void WatchThread();        // Watch CheckFrame thread
        std::string visionsource_ = ""; // cropped image from crop plugin
        std::string tracksource_ = "";  // track result
        uint32_t skipFrameNum_ = 0;
        uint32_t count = 1;
        double timeout_ = 500.; //Millisecond
        uint32_t sleepTime_ = 0;
        std::ostringstream ErrorInfo_;
        std::unique_ptr<std::thread> thread_ = nullptr;
        bool stopFlag_ = false;
        MxTools::InputParam inputParam;      // to create buffer
    };
}
#endif