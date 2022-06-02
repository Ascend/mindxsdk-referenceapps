/*
* Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
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

#include <cstring>
#include <unistd.h>
#include "MxStream/StreamManager/MxStreamManager.h"

namespace {
    const int SIGNAL_CHECK_TIMESTEP = 10000;
    static bool signalRecieved = false;
}

static void SigHandler(int signal)
{
    if (signal == SIGINT) {
        signalRecieved = true;
    }
}

APP_ERROR TestVideoEncoder(std::string &pipelinePath)
{
    // read image file and build stream input
    std::cout << "Begin Load " << pipelinePath << std::endl;
    // init stream manager
    MxStream::MxStreamManager mxStreamManager;
    APP_ERROR ret = mxStreamManager.InitManager();
    if (ret != APP_ERR_OK) {
        std::cout << "Failed to init Stream manager, ret = " << ret << "." << std::endl;
        return ret;
    }
    ret = mxStreamManager.CreateMultipleStreamsFromFile(pipelinePath);
    if (ret != APP_ERR_OK) {
        std::cout << "Failed to create Stream, ret = " << ret << "." << std::endl;
        return ret;
    }

    signal(SIGINT, SigHandler);
    while (!signalRecieved) {
        usleep(SIGNAL_CHECK_TIMESTEP);
    }

    // destroy streams
    mxStreamManager.DestroyAllStreams();
    return APP_ERR_OK;
}

int main(int argc, char* argv[])
{
    std::string pipelinePath(argv[1]);
    TestVideoEncoder(pipelinePath);
    return 0;
}
