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
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cstddef>
#include <Yolov5Detection.h>
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t CLASS_NU = 1;
    const uint32_t BIASES_NU = 18;
    const uint32_t ANCHOR_DIM = 3;
    const uint32_t YOLO_TYPE = 3;
}

void InitYolov5Param(InitParam &initParam)
{
    initParam.deviceId = 0;
    initParam.labelPath = "./model/coco.names";
    initParam.checkTensor = true;
    initParam.modelPath = "./model/onnx_best_v3.om";
    initParam.classNum = CLASS_NU;
    initParam.biasesNum = BIASES_NU;
    initParam.biases = "10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326";
    initParam.objectnessThresh = "0.3";
    initParam.iouThresh = "0.45";
    initParam.scoreThresh = "0.3";
    initParam.yoloType = YOLO_TYPE;
    initParam.modelType = 1;
    initParam.inputType = 0;
    initParam.anchorDim = ANCHOR_DIM;
}

void create_dir(std::string path)
{
    DIR *dir;
    if ((dir = opendir(path.c_str())) == NULL)
    {
        int isCreate = mkdir("./result",S_IRWXU);
        if(!isCreate){
            std::cout << "Create dir success" << std::endl;
        }
        else{
            std::cout << "Create dir failed" << std::endl;  
        }
    }
}
void get_files(std::string path, std::vector<std::string> &files)
{
    DIR *dir = NULL;
    struct dirent *ptr = NULL;

    if ((dir = opendir(path.c_str())) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr = readdir(dir)) != NULL)
    {
        if (ptr->d_type == 8){
            files.push_back(path +"/"+ptr->d_name);
        }
    }
    closedir(dir);
}

int main(int argc, char* argv[])
{
    if (argc <= 1) {
        LogWarn << "Please input folder path, such as './mxBase ./test'.";
        return APP_ERR_OK;
    }
    LogInfo << "Project begin!!!!";
    create_dir("./result");
    std::vector<std::string> files;
    InitParam initParam;
    InitYolov5Param(initParam);
    auto yolov5 = std::make_shared<Yolov5Detection>();
    // 初始化模型推理所需的配置信息
    APP_ERROR ret = yolov5->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Yolov5Detection init failed, ret=" << ret << ".";
        return ret;
    }
    get_files(argv[1], files);
    for(uint32_t i=0;i<files.size();i++)
    {
        // 推理业务开始
        std::string imgPath = files[i];
        ret = yolov5->Process(imgPath);
        if (ret != APP_ERR_OK) {
            LogError << "Yolov5Detection process failed, ret=" << ret << ".";
            yolov5->DeInit();
            return ret;
        }
    }
    yolov5->DeInit();
    LogInfo << "Project end!!!!";
    return APP_ERR_OK;
}