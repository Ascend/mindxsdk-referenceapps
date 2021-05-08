/*
* Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
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
#include <cstring>
#include <unistd.h>
#include "MxBase/Log/Log.h"
#include "MxStream/StreamManager/MxStreamManager.h"

namespace {
bool signalRecieved = false;
}

static void SigHandler(int signal)
{
    if (signal == SIGINT) {
        signalRecieved = true;
    }
}

APP_ERROR TestInferOfflineVideo()
{
    // init stream manager
    MxStream::MxStreamManager mxStreamManager;
    APP_ERROR ret = mxStreamManager.InitManager();
    if (ret != APP_ERR_OK) {
        LogInfo << "Failed to init Stream manager, ret = " << ret << ".";
        return ret;
    }

    // create stream by pipeline config file
    std::string pipelineConfigPath = "./pipeline/regular.pipeline";
    ret = mxStreamManager.CreateMultipleStreamsFromFile(pipelineConfigPath);
    if (ret != APP_ERR_OK) {
        LogInfo << "Failed to create Stream, ret = " << ret << ".";
        return ret;
    }
    signal(SIGINT, SigHandler);

    while (!signalRecieved) {
        MxStream::MxstDataOutput* output = mxStreamManager.GetResult("inferofflinevideo", 0);
        if (output->errorCode != APP_ERR_OK) {
            LogInfo << "Failed to get pipeline output, ret = " << output->errorCode;
            mxStreamManager.DestroyAllStreams();
            return output->errorCode;
        }
        std::string result = std::string((char *)output->dataPtr, output->dataSize);
        LogInfo << "Results:" << result << "\n\n";
    }

    // destroy streams
    mxStreamManager.DestroyAllStreams();
    return APP_ERR_OK;
}

int main(int argc, char* argv[])
{
    TestInferOfflineVideo();
    return 0;
}
