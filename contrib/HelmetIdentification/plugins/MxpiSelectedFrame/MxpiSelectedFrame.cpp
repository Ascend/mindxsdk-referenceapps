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

#include "./ExampleProject/HelmetIdentification/plugins/MxpiSelectedFrame/MxpiSelectedFrame.h"
#include "./MindX_SDK/mxVision/include/MxBase/Log/Log.h"
using namespace MxBase;
using namespace MxTools;
using namespace MxPlugins;

APP_ERROR MxpiSelectedFrame::Init(std::map<std::string, std::shared_ptr<void>> &configParamMap)
{
    LogInfo << "Begin to initialize MxpiSelectedFrame(" << pluginName_ << ").";
    // get parameters from website.
    SelectedFrameNum_ = *std::static_pointer_cast<uint>(configParamMap["frameNum"]);
    LogInfo << "skip frame nunmber(" << SelectedFrameNum_ << ").";
    LogInfo << "End to initialize MxpiSelectedFrame(" << pluginName_ << ").";
    return APP_ERR_OK;
}

APP_ERROR MxpiSelectedFrame::DeInit()
{
    LogInfo << "Begin to deinitialize MxpiSelectedFrame(" << pluginName_ << ").";
    LogInfo << "End to deinitialize MxpiSelectedFrame(" << pluginName_ << ").";
    return APP_ERR_OK;
}

APP_ERROR MxpiSelectedFrame::Process(std::vector<MxpiBuffer *> &mxpiBuffer)
{
    MxpiBuffer *inputMxpiBuffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*inputMxpiBuffer);
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr!=nullptr) {
        LogWarn << "Input data is invalid, element(" << pluginName_ << ") plugin will not be executed rightly.";
        SendData(0, *inputMxpiBuffer);
        return APP_ERR_COMM_FAILURE;
    }

    if (SelectedFrameNum_ == 0) {
        SendData(0, *inputMxpiBuffer);
    } else {
        count++;
        if ((count % (SelectedFrameNum_ + 1)) == 0) {
            count = 0;
            SendData(0, *inputMxpiBuffer);
        } else {
            MxpiBufferManager::DestroyBuffer(inputMxpiBuffer);
        }
    }
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiSelectedFrame::DefineProperties()
{
    std::vector<std::shared_ptr<void>> properties;
    auto prop1 = std::make_shared<ElementProperty<uint>>(ElementProperty<uint> {
            UINT,
            "frameNum",
            "frameNum",
            "the number of selected frame",
            0, 0, 100
    });
    properties.push_back(prop1);
    return properties;
}

MxpiPortInfo MxpiSelectedFrame::DefineInputPorts()
{
    MxpiPortInfo inputPortInfo;
    std::vector<std::vector<std::string>> value = {{"image/yuv"}};
    GenerateStaticInputPortsInfo(value, inputPortInfo);
    return inputPortInfo;
}

MxpiPortInfo MxpiSelectedFrame::DefineOutputPorts()
{
    MxpiPortInfo outputPortInfo;
    std::vector<std::vector<std::string>> value = {{"image/yuv"}};
    GenerateStaticOutputPortsInfo(value, outputPortInfo);
    return outputPortInfo;
}
namespace {
    MX_PLUGIN_GENERATE(MxpiSelectedFrame)
}