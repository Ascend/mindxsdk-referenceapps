/*
 * Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

#include <iostream>
#include <vector>
#include "MediaCodecV2.h"
#include "MxBase/Log/Log.h"

const uint32_t ARGC_NUMBER = 2;
bool MediaCodecv2::stopFlag = false;
std::shared_ptr<MediaCodecv2> g_p = nullptr;

static void my_handler(int sig) {
    if (g_p != nullptr) {
        LogInfo << "in handle";
        g_p->stopProcess();
    }
}

int main(int argc, char *argv[])
{
    if (argc <= 1) {
        LogWarn << "Please input image path, such as '../../data/video_test.264'.";
        return APP_ERR_OK;
    } else if (argc > 1 && argc <= ARGC_NUMBER) {
        LogWarn << "Please input output image path, such as '../out/out_test.264'.";
        return APP_ERR_OK;
    }

    APP_ERROR ret;
    // global init
    ret = MxBase::MxInit();
    if (ret != APP_ERR_OK)
    {
        LogError << "MxInit failed, ret=" << ret << ".";
    }

    auto mediacodecv2 = std::make_shared<MediaCodecv2>();
    std::string filePath = argv[1];
    LogInfo << "filePath = " << filePath;

    std::string outPath = argv[2];
    LogInfo << "outPath = " << outPath;

    ret = mediacodecv2->Init(filePath, outPath);
    if (ret != APP_ERR_OK) {
        LogError << "MediaCodecv2 process failed, ret=" << ret << ".";
        return ret;
    }

    g_p = mediacodecv2;
    signal(SIGINT, my_handler);

    ret = mediacodecv2->Process(filePath, outPath);
    if (ret != APP_ERR_OK) {
        LogError << "MediaCodecv2 process failed, ret=" << ret << ".";
        return ret;
    }

    LogInfo << "ALL DONE";
    return APP_ERR_OK;
}
