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

#include <iostream>
#include <vector>
#include <RcfDetection.h>
#include <boost/filesystem.hpp>
#include "MxBase/Log/Log.h"

namespace fs = boost::filesystem;
namespace {
    const uint32_t OUTSIZE_NU = 5;
    const uint32_t RCF_TYPE = 5;
}

static void InitRcfParam(InitParam &initParam)
{
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "./model/rcf.om";
    initParam.outSizeNum = OUTSIZE_NU;
    initParam.rcfType = RCF_TYPE;
    initParam.modelType = 0;
    initParam.inputType = 0;
    // model output feature size
    initParam.outSize = "512,256,128,64,63";
}

int main(int argc, char *argv[])
{
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './test.jpg'.";
        return APP_ERR_OK;
    }
    InitParam initParam;
    InitRcfParam(initParam);
    auto rcf = std::make_shared<RcfDetection>();
    APP_ERROR ret = rcf->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "rcfDetection init failed, ret=" << ret << ".";
        return ret;
    }
    std::string imgPath = argv[1];
    for (auto & entry : fs::directory_iterator(imgPath)) {
        LogInfo << "read image path " << entry.path();
        ret = rcf->Process(entry.path().string());
        if (ret != APP_ERR_OK) {
            LogError << "Inceptionv4Opencv process failed, ret=" << ret << ".";
            rcf->DeInit();
            return ret;
        }
    }

    rcf->DeInit();
    return APP_ERR_OK;
}
