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

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <algorithm>
#include <map>

#include <opencv2/opencv.hpp>
#include "MxBase/MxBase.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "PPYoloePostProcess.h"

using namespace MxBase;
using namespace std;

const int MODEL_INPUT_WIDTH = 640;
const int MODEL_INPUT_HEIGHT = 640;
const int AVG_PARAM = 2;
const long MAX_FILE_SIZE = 1024 * 1024 * 1024; // 1g

APP_ERROR CheckFileVaild(const std::string &filePath)
{
    struct stat buf;
    if (lstat(filePath.c_str(), &buf) != 0 || S_ISLNK(buf.st_mode)) {
        LogError << "Input file is invalid and cannot be a link";
        return APP_ERR_COMM_NO_EXIST;
    }
    char c[PATH_MAX + 1] = {0x00};
    size_t count = filePath.copy(c, PATH_MAX + 1);
    if (count != filePath.length()) {
        LogError << "Failed to copy file path.";
        return APP_ERR_COMM_FAILURE;
    }
    char path[PATH_MAX + 1] = {0x00};
    if (realpath(c, path) == nullptr) {
        LogError << "Failed to get the file.";
        return APP_ERR_COMM_NO_EXIST;
    }
    FILE *fp = fopen(path, "rb");
    if (fp == nullptr) {
        LogError << "Failed to open file";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fileSize <= 0 || fileSize > MAX_FILE_SIZE) {
        fclose(fp);
        return APP_ERR_COMM_FAILURE;
    }
    fclose(fp);
    return APP_ERR_OK;
}

std::vector<std::vector<ObjectInfo>> SDKPostProcess(std::string &ppyoloeConfigPath, std::string &ppyoloeLabelPath,
    std::vector<Tensor> &ppyoloeOutputs, std::vector<ResizedImageInfo> &imagePreProcessInfos)
{
    std::map<std::string, std::string> postConfig;
    postConfig.insert(pair<std::string, std::string>("postProcessConfigPath", ppyoloeConfigPath));
    postConfig.insert(pair<std::string, std::string>("labelPath", ppyoloeLabelPath));

    PPYoloePostProcess ppyoloePostProcess;
    ppyoloePostProcess.Init(postConfig);
    
    std::vector<TensorBase> tensors;
    for (size_t i = 0; i < ppyoloeOutputs.size(); i++) {
        MemoryData memoryData(ppyoloeOutputs[i].GetData(), ppyoloeOutputs[i].GetByteSize());
        TensorBase tensorBase(memoryData, true, ppyoloeOutputs[i].GetShape(), TENSOR_DTYPE_INT32);
        tensors.push_back(tensorBase);
    }
    std::vector<std::vector<ObjectInfo>> objectInfos;
    ppyoloePostProcess.Process(tensors, objectInfos, imagePreProcessInfos);
    for (size_t i = 0; i < objectInfos.size(); i++) {
        LogInfo << "objectInfos-" << i;
        for (size_t j = 0; j < objectInfos[i].size(); j++) {
            LogInfo << " objectInfo-" << j;
            LogInfo << "      x0 is:" << objectInfos[i][j].x0;
            LogInfo << "      y0 is:" << objectInfos[i][j].y0;
            LogInfo << "      x1 is:" << objectInfos[i][j].x1;
            LogInfo << "      y1 is:" << objectInfos[i][j].y1;
            LogInfo << "      confidence is: " << objectInfos[i][j].confidence;
            LogInfo << "      classId is: " << objectInfos[i][j].classId;
            LogInfo << "      className is: " << objectInfos[i][j].className;
        }
    }
    return objectInfos;
}

APP_ERROR DvppPreprocessorYuv(ImageProcessor &imageProcessor, std::string &imagePath, vector<Tensor> &ppyoloeInputs,
    std::vector<ResizedImageInfo> &imagePreProcessInfos, int deviceId)
{
    Image decodeImage;
    APP_ERROR ret = imageProcessor.Decode(imagePath, decodeImage);
    if (ret != APP_ERR_OK) {
        LogError << "ImageProcessor decode failed.";
        return ret;
    }
    Image resizeImage;
    uint32_t originalWidth = decodeImage.GetOriginalSize().width;
    uint32_t originalHeight = decodeImage.GetOriginalSize().height;
    ret = imageProcessor.Resize(decodeImage, MxBase::Size(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT), resizeImage,
        Interpolation::BILINEAR_SIMILAR_OPENCV);
    if (ret != APP_ERR_OK) {
        LogError << "ImageProcessor resize failed.";
        return ret;
    }
    ppyoloeInputs.push_back(resizeImage.ConvertToTensor());
    ResizedImageInfo imagePreProcessInfo(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, originalWidth, originalHeight,
        RESIZER_STRETCHING, 0);
    imagePreProcessInfos.push_back(imagePreProcessInfo);
    return APP_ERR_OK;
}

APP_ERROR DvppPreprocessor(std::string &imagePath, vector<Tensor> &ppyoloeInputs,
    std::vector<ResizedImageInfo> &imagePreProcessInfos, int deviceId)
{
    ImageProcessor imageProcessor(deviceId);
    if (DeviceManager::IsAscend310P()) {
        Image decodeImage;
        APP_ERROR ret = imageProcessor.Decode(imagePath, decodeImage, ImageFormat::BGR_888);
        if (ret != APP_ERR_OK) {
            LogError << "ImageProcessor decode failed.";
            return ret;
        }
        Image resizeImage;
        uint32_t originalWidth = decodeImage.GetOriginalSize().width;
        uint32_t originalHeight = decodeImage.GetOriginalSize().height;
        ret = imageProcessor.Resize(decodeImage, MxBase::Size(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT), resizeImage,
            Interpolation::BILINEAR_SIMILAR_OPENCV);
        if (ret != APP_ERR_OK) {
            LogError << "ImageProcessor resize failed.";
            return ret;
        }
        ppyoloeInputs.push_back(resizeImage.ConvertToTensor());
        ResizedImageInfo imagePreProcessInfo(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, originalWidth, originalHeight,
            RESIZER_STRETCHING, 0);
        imagePreProcessInfos.push_back(imagePreProcessInfo);
    } else {
        return DvppPreprocessorYuv(imageProcessor, imagePath, ppyoloeInputs, imagePreProcessInfos, deviceId);
    }
    return APP_ERR_OK;
}

APP_ERROR E2eInfer(std::string imagePath, int32_t deviceId)
{
    vector<Tensor> ppyoloeInputs;
    std::vector<ResizedImageInfo> resizedImageInfos;
    APP_ERROR ret = DvppPreprocessor(imagePath, ppyoloeInputs, resizedImageInfos, deviceId);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    string modelPath = "model/ppyoloe.om";
    ret = CheckFileVaild(modelPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    Model ppyoloe(modelPath, deviceId);

    vector<Tensor> ppyoloeOutputs = ppyoloe.Infer(ppyoloeInputs);
    if (ppyoloeInputs.size() == 0) {
        LogError << "PPYOLOE infer failed.";
        return APP_ERR_COMM_FAILURE;
    }
    for (size_t i = 0; i < ppyoloeOutputs.size(); i++) {
        ppyoloeOutputs[i].ToHost();
    }
    std::vector<Rect> cropConfigVec;
    string ppyoloeConfigPath = "model/ppyoloe.cfg";
    string ppyoloeLabelPath = "model/ppyoloe.names";

    ret = CheckFileVaild(ppyoloeConfigPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    ret = CheckFileVaild(ppyoloeLabelPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    std::vector<std::vector<ObjectInfo>> objectInfos =
        SDKPostProcess(ppyoloeConfigPath, ppyoloeLabelPath, ppyoloeOutputs, resizedImageInfos);
    return APP_ERR_OK;
}

int main(int argc, char *argv[])
{
    MxInit();
    int32_t deviceId = 0;
    std::string imgPath = "test.jpg";
    APP_ERROR ret = CheckFileVaild(imgPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    ret = E2eInfer(imgPath, deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to run E2eInfer";
    }
    return 0;
}