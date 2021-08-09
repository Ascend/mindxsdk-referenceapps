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
#include <CrowdCount.h>
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;

namespace {
    const uint32_t CLASS_NU = 1;
}

void InitCrowdCountParam(InitParam &initParam)
{
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "./model/count_person.caffe.om";
    initParam.classNum = CLASS_NU;
    initParam.labelPath = "./model/count_person.names";
}

int main(int argc, char* argv[])
{
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './CrowdCountPostProcess crowd.jpg'.";
        return APP_ERR_OK;
    }

    InitParam initParam;
    InitCrowdCountParam(initParam);
    auto crowdcount = std::make_shared<CrowdCountOpencv>();
    APP_ERROR ret = crowdcount->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "CrowdCount init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    ret = crowdcount->Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "CrowdCount process failed, ret=" << ret << ".";
        crowdcount->DeInit();
        return ret;
    }
    crowdcount->DeInit();
    return APP_ERR_OK;
}
