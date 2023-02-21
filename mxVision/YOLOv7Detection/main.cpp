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
#include "Yolov7PostProcess.h"

using namespace MxBase;
using namespace std;

const int MODEL_INPUT_WIDTH = 640;
const int MODEL_INPUT_HEIGHT = 640;
const int RGB_EXTEND = 3;
const int PAD_COLOR = 114;
const int OPENCV_8UC3 = 16;
const int YUV_DIVISION = 2;
const int R_CHANNEL = 2;
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

std::vector<std::vector<ObjectInfo>> SDKPostProcess(std::string &yolov7ConfigPath, std::string &yolov7LabelPath,
    std::vector<Tensor> &yolov7Outputs, std::vector<ResizedImageInfo> &imagePreProcessInfos)
{
    std::map<std::string, std::string> postConfig;
    postConfig.insert(pair<std::string, std::string>("postProcessConfigPath", yolov7ConfigPath));
    postConfig.insert(pair<std::string, std::string>("labelPath", yolov7LabelPath));

    Yolov7PostProcess yolov7PostProcess;
    yolov7PostProcess.Init(postConfig);

    std::vector<TensorBase> tensors;
    for (size_t i = 0; i < yolov7Outputs.size(); i++) {
        MemoryData memoryData(yolov7Outputs[i].GetData(), yolov7Outputs[i].GetByteSize());
        TensorBase tensorBase(memoryData, true, yolov7Outputs[i].GetShape(), TENSOR_DTYPE_INT32);
        tensors.push_back(tensorBase);
    }
    std::vector<std::vector<ObjectInfo>> objectInfos;
    yolov7PostProcess.Process(tensors, objectInfos, imagePreProcessInfos);
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

APP_ERROR PaddingProcess(ImageProcessor &imageProcessor, std::pair<int, int> resizeInfo, int deviceId,
    Image &resizeImage, Image &pastedImg)
{
    int resizedWidth = resizeInfo.first;
    int resizedHeight = resizeInfo.second;
    int leftOffset = (MODEL_INPUT_WIDTH - resizedWidth) / AVG_PARAM;
    int topOffset = (MODEL_INPUT_HEIGHT - resizedHeight) / AVG_PARAM;
    uint32_t dataSize = MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * RGB_EXTEND;
    MxBase::Size imageSize(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);
    if (leftOffset > 0) {
        MemoryData srcData(resizeImage.GetData().get(), resizeImage.GetDataSize(), MemoryData::MemoryType::MEMORY_DVPP,
            deviceId);
        MemoryData resHostData(nullptr, resizeImage.GetDataSize(), MemoryData::MemoryType::MEMORY_HOST, -1);
        if (MemoryHelper::MxbsMallocAndCopy(resHostData, srcData) != APP_ERR_OK) {
            LogError << "Failed to mallloc and copy dvpp memory.";
            return APP_ERR_ACL_BAD_COPY;
        }
        cv::Mat resizedHost(resizeImage.GetSize().height, resizeImage.GetSize().width, OPENCV_8UC3, resHostData.ptrData);
        cv::Rect roi = cv::Rect(0, 0, resizedWidth, resizedHeight);
        cv::Mat extendedImage;
        cv::copyMakeBorder(resizedHost(roi), extendedImage, 0, 0, leftOffset, MODEL_INPUT_WIDTH - leftOffset - resizedWidth,
            cv::BORDER_CONSTANT, cv::Scalar(PAD_COLOR, PAD_COLOR, PAD_COLOR));
        int maxFillRow = std::min(MODEL_INPUT_WIDTH, (int)resizeImage.GetSize().width + leftOffset);
        for (int col = 0; col < MODEL_INPUT_WIDTH; col++) {
            for (int row = resizedWidth + leftOffset; row < maxFillRow; row++) {
                extendedImage.at<cv::Vec3b>(col, row)[0] = PAD_COLOR;
                extendedImage.at<cv::Vec3b>(col, row)[1] = PAD_COLOR;
                extendedImage.at<cv::Vec3b>(col, row)[R_CHANNEL] = PAD_COLOR;
            }
        }
        uint8_t* pasteHostData = (uint8_t*)malloc(dataSize);
        if (pasteHostData == nullptr) {
            return APP_ERR_ACL_BAD_ALLOC;
        }
        for (size_t i = 0; i < dataSize; i++) {
            pasteHostData[i] = extendedImage.data[i];
        }
        std::shared_ptr<uint8_t> dataPaste((uint8_t*)pasteHostData, free);
        Image pastedImgTmp(dataPaste, dataSize, -1, imageSize, ImageFormat::BGR_888);
        pastedImgTmp.ToDevice(0);
        pastedImg = pastedImgTmp;
    } else {
        MemoryData imgData(dataSize, MemoryData::MemoryType::MEMORY_DVPP, deviceId);
        if (MemoryHelper::Malloc(imgData) != APP_ERR_OK) {
            return APP_ERR_ACL_BAD_ALLOC;
        }
        std::shared_ptr<uint8_t> pastedData((uint8_t*)imgData.ptrData, imgData.free);
        if (MemoryHelper::Memset(imgData, PAD_COLOR, dataSize) != APP_ERR_OK) {
            LogError << "Failed to memset dvpp memory.";
            return APP_ERR_ACL_BAD_ALLOC;
        }
        Rect RectSrc(0, 0, resizedWidth, resizedHeight);
        Rect RectDst(leftOffset, topOffset, leftOffset + resizedWidth, topOffset + resizedHeight);
        std::pair<Rect, Rect> cropPasteRect = {RectSrc, RectDst};
        Image pastedImgTmp(pastedData, dataSize, deviceId, imageSize, ImageFormat::BGR_888);
        if (imageProcessor.CropAndPaste(resizeImage, cropPasteRect, pastedImgTmp) != APP_ERR_OK) {
            LogError << "Failed to padding the image by dvpp";
            return APP_ERR_COMM_FAILURE;
        }
        pastedImg = pastedImgTmp;
    }
    return APP_ERR_OK;
}

APP_ERROR DvppPreprocessorYuv(ImageProcessor &imageProcessor, std::string &imagePath, vector<Tensor> &yolov7Inputs,
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
    float scaleWidth = MODEL_INPUT_WIDTH * 1.0 / originalWidth;
    float scaleHeight = MODEL_INPUT_HEIGHT * 1.0 / originalHeight;
    float minScale = scaleWidth < scaleHeight ? scaleWidth : scaleHeight;
    int resizedWidth = std::round(originalWidth * minScale);
    int resizedHeight = std::round(originalHeight * minScale);
    ret = imageProcessor.Resize(decodeImage, MxBase::Size(resizedWidth, resizedHeight), resizeImage,
        Interpolation::BILINEAR_SIMILAR_OPENCV);
    if (ret != APP_ERR_OK) {
        LogError << "ImageProcessor resize failed.";
        return ret;
    }
    uint32_t dataSize = MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * RGB_EXTEND / YUV_DIVISION;
    MxBase::Size imageSize(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);
    MemoryData imgData(dataSize, MemoryData::MemoryType::MEMORY_DVPP, deviceId);
    if (MemoryHelper::Malloc(imgData) != APP_ERR_OK) {
        LogError << "Failed to malloc dvpp memory.";
        return APP_ERR_ACL_BAD_ALLOC;
    }
    std::shared_ptr<uint8_t> pastedData((uint8_t*)imgData.ptrData, imgData.free);
    if (MemoryHelper::Memset(imgData, PAD_COLOR, dataSize) != APP_ERR_OK) {
        LogError << "Failed to memset dvpp memory.";
        return APP_ERR_ACL_BAD_ALLOC;
    }
    int leftOffset = (MODEL_INPUT_WIDTH - resizedWidth) / AVG_PARAM;
    int topOffset = (MODEL_INPUT_HEIGHT - resizedHeight) / AVG_PARAM;
    Rect RectSrc(0, 0, resizedWidth, resizedHeight);
    Rect RectDst(leftOffset, topOffset, leftOffset + resizedWidth, topOffset + resizedHeight);
    std::pair<Rect, Rect> cropPasteRect = {RectSrc, RectDst};
    Image pastedImgTmp(pastedData, dataSize, deviceId, imageSize, ImageFormat::YUV_SP_420);
    if (imageProcessor.CropAndPaste(resizeImage, cropPasteRect, pastedImgTmp) != APP_ERR_OK) {
        LogError << "Failed to padding the image by dvpp";
        return APP_ERR_COMM_FAILURE;
    }
    yolov7Inputs.push_back(pastedImgTmp.ConvertToTensor());
    ResizedImageInfo imagePreProcessInfo(resizedWidth, resizedHeight, originalWidth, originalHeight,
        RESIZER_MS_KEEP_ASPECT_RATIO, minScale);
    imagePreProcessInfos.push_back(imagePreProcessInfo);
    return APP_ERR_OK;
}

APP_ERROR DvppPreprocessor(std::string &imagePath, vector<Tensor> &yolov7Inputs,
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
        float scaleWidth = MODEL_INPUT_WIDTH * 1.0 / originalWidth;
        float scaleHeight = MODEL_INPUT_HEIGHT * 1.0 / originalHeight;
        float minScale = scaleWidth < scaleHeight ? scaleWidth : scaleHeight;
        int resizedWidth = std::round(originalWidth * minScale);
        int resizedHeight = std::round(originalHeight * minScale);
        ret = imageProcessor.Resize(decodeImage, MxBase::Size(resizedWidth, resizedHeight), resizeImage,
            Interpolation::BILINEAR_SIMILAR_OPENCV);
        if (ret != APP_ERR_OK) {
            LogError << "ImageProcessor resize failed.";
            return ret;
        }
        Image pastedImage;
        std::pair<int, int> resizedInfo(resizedWidth, resizedHeight);
        ret = PaddingProcess(imageProcessor, resizedInfo, deviceId, resizeImage, pastedImage);
        if (ret != APP_ERR_OK) {
            LogError << "ImageProcessor padding failed.";
            return ret;
        }
        yolov7Inputs.push_back(pastedImage.ConvertToTensor());
        ResizedImageInfo imagePreProcessInfo(resizedWidth, resizedHeight, originalWidth, originalHeight,
            RESIZER_STRETCHING, 0);
        imagePreProcessInfos.push_back(imagePreProcessInfo);
    } else {
        return DvppPreprocessorYuv(imageProcessor, imagePath, yolov7Inputs, imagePreProcessInfos, deviceId);
    }
    return APP_ERR_OK;
}

APP_ERROR E2eInfer(std::string imagePath, int32_t deviceId)
{
    vector<Tensor> yolov7Inputs;
    std::vector<ResizedImageInfo> resizedImageInfos;
    APP_ERROR ret = DvppPreprocessor(imagePath, yolov7Inputs, resizedImageInfos, deviceId);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    string modelPath = "model/yolov7.om";
    ret = CheckFileVaild(modelPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    Model yolov7(modelPath, deviceId);

    vector<Tensor> yolov7Outputs = yolov7.Infer(yolov7Inputs);
    if (yolov7Inputs.size() == 0) {
        LogError << "YOLOv7 infer failed.";
        return APP_ERR_COMM_FAILURE;
    }
    for (size_t i = 0; i < yolov7Outputs.size(); i++) {
        yolov7Outputs[i].ToHost();
    }
    std::vector<Rect> cropConfigVec;
    string yolov7ConfigPath = "model/yolov7.cfg";
    string yolov7LabelPath = "model/coco.names";

    ret = CheckFileVaild(yolov7ConfigPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    ret = CheckFileVaild(yolov7LabelPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    std::vector<std::vector<ObjectInfo>> objectInfos =
        SDKPostProcess(yolov7ConfigPath, yolov7LabelPath, yolov7Outputs, resizedImageInfos);
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