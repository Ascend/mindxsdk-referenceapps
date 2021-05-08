/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <MxTools/PluginToolkit/buffer/MxpiBufferManager.h>
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/CV/Core/DataType.h"
#include "MxpiAllObjectsStructuringDataType.pb.h"
#include "BlockingMap.h"

#ifndef MXPLUGINS_MXPIFRAMEALIGN_H
#define MXPLUGINS_MXPIFRAMEALIGN_H

struct StreamData {
    MxpiWebDisplayData webDisplayData = {};
    bool sendFlag = false;
};

struct ObjectInfo {
    std::string trackId;
    float x0;
    float y0;
    float x1;
    float y1;
};

class MxpiFrameAlign : public MxTools::MxPluginBase {
public:
    /**
     * @api
     * @param configParamMap
     * @return
     */
    APP_ERROR Init(std::map<std::string, std::shared_ptr<void>> &configParamMap) override;

    /**
     * @api
     * @return
     */
    APP_ERROR DeInit() override;

    /**
     * @api
     * @param mxpiBuffer
     * @return
     */
    APP_ERROR Process(std::vector<MxTools::MxpiBuffer*> &mxpiBuffer) override;

    /**
     * @api
     * @brief Definition the parameter of configure properties.
     * @return std::vector<std::shared_ptr<void>>
     */
    static std::vector<std::shared_ptr<void>> DefineProperties();

    /**
     * Optional, defines input ports of the plugin.
     */
    static MxTools::MxpiPortInfo DefineInputPorts();

    /**
     * Optional, defines output ports of the plugin.
     */
    static MxTools::MxpiPortInfo DefineOutputPorts();

private:
    void GetStreamData(MxTools::MxpiBuffer &inputBuffer);

    APP_ERROR GetObjectList(MxTools::MxpiBuffer &inputBuffer);

    void AlignFrameObjectInfo();

    bool HadTrackId(std::vector<ObjectInfo> &objectInfoList, std::string &trackId);

    void ObjectInfoInterpolated(std::vector<ObjectInfo> &interpolatedObjectInfoList,
                                std::vector<ObjectInfo> &previousObjectInfoList,
                                std::vector<ObjectInfo> &latterObjectInfoList,
                                float &offset);

    APP_ERROR SendAlignFrame();

    APP_ERROR GetWebDisplayData(std::shared_ptr<MxpiWebDisplayDataList> &webDisplayDataList, uint32_t &frameId);

    std::vector<std::string> Split(const std::string &inString, char delimiter = ' ');

    std::string &Trim(std::string &str);

    std::vector<std::string> SplitWithRemoveBlank(std::string &str, char rule);

    void SendThread();

private:
    std::ostringstream errorInfo_ {};
    std::vector<std::string> dataKeyVec_ = {};
    std::string dataSource_ = "";
    uint32_t previousFrameId_ = 0;
    bool sendStop_ = false;
    std::thread sendThread_ = {};
    int intervalTime_ = 0;
    BlockingMap<uint32_t, StreamData> streamDataMap_ = {};
    BlockingMap<uint32_t, std::vector<ObjectInfo>> objectListMap_ = {};
};

#endif
