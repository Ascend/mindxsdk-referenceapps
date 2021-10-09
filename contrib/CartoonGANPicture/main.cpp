/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

#include <iostream>
#include <vector>
#include "boost/filesystem.hpp"
#include "CartoonGANPicture/CartoonGANPicture.h"

std::vector<double> g_inferCost;
namespace fs = boost::filesystem;

APP_ERROR ReadImagesPath(const std::string &imgPath, std::vector<std::string> &imagesPath)
{
    if(!fs::exists(imgPath) )
    {
        LogError << " directory is not exist." ;
        return APP_ERR_COMM_FAILURE;
    }
    fs::directory_iterator item_begin(imgPath);
    fs::directory_iterator item_end;
    if (item_begin == item_end)
    {
        LogError << " directory is null.";
        return APP_ERR_COMM_FAILURE;
    }

    for (auto & entry : fs::directory_iterator(imgPath))
    {
        imagesPath.push_back(entry.path().string());
    }

    return APP_ERR_OK;
}

void InitYolov3Param(InitParam &initParam)
{
    initParam.deviceId = 0;
    initParam.modelPath = "./data/model/cartoonization.om";
}

int main(int argc, char* argv[])
{
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './CartoonGAN_picture ./data/images'.";
        return APP_ERR_OK;
    }

    InitParam initParam;
    InitYolov3Param(initParam);
    auto cartoon = std::make_shared<CartoonGANPicture>();
    APP_ERROR ret = cartoon->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "CartoonGANPicture init failed, ret=" << ret << ".";
        return ret;
    }
    //std::vector<double> g_inferCost;
    std::string inferText = argv[1];
    std::vector<std::string> imagesPath;
    ret = ReadImagesPath(inferText, imagesPath);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImagesPath failed, ret=" << ret << ".";
        cartoon->DeInit();
        return ret;
    }
    for (uint32_t i = 0; i < imagesPath.size(); i++) {
        LogInfo << imagesPath[i];
        auto startTime = std::chrono::high_resolution_clock::now();
        ret = cartoon->Process(imagesPath[i]);
        auto endTime = std::chrono::high_resolution_clock::now();
        double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        if (ret != APP_ERR_OK) {
            LogError << "CartoonGANPicture process failed, ret=" << ret << ".";
            continue;
        }
        g_inferCost.push_back(costMs);
    }

    double costSum = 0;
    for (uint32_t i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "Infer average time " << costSum / g_inferCost.size() << " ms.";

    cartoon->DeInit();
    return APP_ERR_OK;
}