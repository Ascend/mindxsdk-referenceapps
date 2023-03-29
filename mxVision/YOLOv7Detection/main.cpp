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
#include "opencv2/imgproc.hpp"
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
const int ARG_NUM = 10;
const long MAX_FILE_SIZE = 1024 * 1024 * 1024; // 1g

const float YUV_Y_R = 0.299;
const float YUV_Y_G = 0.587;
const float YUV_Y_B = 0.114;
const float YUV_U_R = -0.169;
const float YUV_U_G = 0.331;
const float YUV_U_B = 0.500;
const float YUV_V_R = 0.500;
const float YUV_V_G = 0.419;
const float YUV_V_B = 0.081;
const int YUV_DATA_SIZE = 3;
const int YUV_OFFSET = 2;
const int YUV_OFFSET_S = 1;
const int YUV_OFFSET_UV = 128;
const int ALIGN_LEFT = 16;
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
        cv::Mat resizedHost(resizeImage.GetSize().height, resizeImage.GetSize().width, OPENCV_8UC3,
            resHostData.ptrData);
        cv::Rect roi = cv::Rect(0, 0, resizedWidth, resizedHeight);
        cv::Mat extendedImage;
        cv::copyMakeBorder(resizedHost(roi), extendedImage, 0, 0, leftOffset,
            MODEL_INPUT_WIDTH - leftOffset - resizedWidth, cv::BORDER_CONSTANT,
            cv::Scalar(PAD_COLOR, PAD_COLOR, PAD_COLOR));
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

APP_ERROR SetImageBackground(MxBase::MemoryData& data)
{
    auto dataPtr = data.ptrData;
    float yuvY = YUV_Y_R * PAD_COLOR + YUV_Y_G * PAD_COLOR + YUV_Y_B * PAD_COLOR;
    float yuvU = YUV_U_R * PAD_COLOR - YUV_U_G * PAD_COLOR + YUV_U_B * PAD_COLOR + YUV_OFFSET_UV;
    float yuvV = YUV_V_R * PAD_COLOR - YUV_V_G * PAD_COLOR - YUV_V_B * PAD_COLOR + YUV_OFFSET_UV;

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMemset(data, (int)yuvY, data.size);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to memset dvpp memory";
        return ret;
    }
    int offsetSize = MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH / YUV_OFFSET;
    data.ptrData = (uint8_t *)data.ptrData + MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH;
    ret = MxBase::MemoryHelper::MxbsMemset(data, (int)yuvU, offsetSize);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to memset dvpp memory";
        data.ptrData = dataPtr;
        return ret;
    }
    data.ptrData = (uint8_t *)data.ptrData + YUV_OFFSET_S;
    for (int i = 0; i < offsetSize / YUV_OFFSET; i++) {
        ret = MxBase::MemoryHelper::MxbsMemset(data, (int)yuvV, YUV_OFFSET_S);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to memset dvpp memory";
            data.ptrData = dataPtr;
            return ret;
        }
        data.ptrData = (uint8_t *)data.ptrData + YUV_OFFSET;
    }
    data.ptrData = dataPtr;
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
    if (SetImageBackground(imgData) != APP_ERR_OK) {
        LogError << "Failed to memset dvpp memory.";
        return APP_ERR_ACL_BAD_ALLOC;
    }
    int leftOffset = (MODEL_INPUT_WIDTH - resizedWidth) / AVG_PARAM;
    int topOffset = (MODEL_INPUT_HEIGHT - resizedHeight) / AVG_PARAM;
    topOffset = topOffset % AVG_PARAM == 0 ? topOffset : topOffset - 1;
    leftOffset = leftOffset < ALIGN_LEFT ? 0 : leftOffset / ALIGN_LEFT * ALIGN_LEFT;
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

APP_ERROR OpenCVPreProcessor(std::string &imagePath, vector<Tensor> &yolov7Inputs,
    std::vector<ResizedImageInfo> &imagePreProcessInfos, int deviceId)
{
    auto image = cv::imread(imagePath);
    size_t originalWidth = image.cols;
    size_t originalHeight = image.rows;
    float scaleWidth = MODEL_INPUT_WIDTH * 1.0 / originalWidth;
    float scaleHeight = MODEL_INPUT_HEIGHT * 1.0 / originalHeight;
    float minScale = scaleWidth < scaleHeight ? scaleWidth : scaleHeight;
    int resizedWidth = std::round(originalWidth * minScale);
    int resizedHeight = std::round(originalHeight * minScale);
    cv::Mat resizedImg;
    cv::resize(image, resizedImg, cv::Size(resizedWidth, resizedHeight));
    int leftOffset = (MODEL_INPUT_WIDTH - resizedWidth) / AVG_PARAM;
    int topOffset = (MODEL_INPUT_HEIGHT - resizedHeight) / AVG_PARAM;
    uint32_t dataSize = MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH * RGB_EXTEND;
    MxBase::Size imageSize(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);
    cv::Mat extendedImage;
    cv::copyMakeBorder(resizedImg, extendedImage, topOffset, MODEL_INPUT_HEIGHT - topOffset - resizedHeight, leftOffset,
        MODEL_INPUT_WIDTH - leftOffset - resizedWidth, cv::BORDER_CONSTANT,
        cv::Scalar(PAD_COLOR, PAD_COLOR, PAD_COLOR));
    uint8_t *pasteHostData = (uint8_t *)malloc(dataSize);
    if (pasteHostData == nullptr) {
        return APP_ERR_ACL_BAD_ALLOC;
    }
    for (size_t i = 0; i < dataSize; i++) {
        pasteHostData[i] = extendedImage.data[i];
    }
    std::shared_ptr<uint8_t> dataPaste((uint8_t *)pasteHostData, free);
    Image pastedImage(dataPaste, dataSize, -1, imageSize, ImageFormat::BGR_888);
    pastedImage.ToDevice(deviceId);
    yolov7Inputs.push_back(pastedImage.ConvertToTensor());
    ResizedImageInfo imagePreProcessInfo(resizedWidth, resizedHeight, originalWidth, originalHeight,
        RESIZER_TF_KEEP_ASPECT_RATIO, minScale);
    imagePreProcessInfos.push_back(imagePreProcessInfo);
    return APP_ERR_OK;
}

APP_ERROR DvppPreprocessor(std::string &imagePath, vector<Tensor> &yolov7Inputs,
    std::vector<ResizedImageInfo> &imagePreProcessInfos, int deviceId, bool isYuvInput)
{
    ImageProcessor imageProcessor(deviceId);
    if (isYuvInput) {
        return DvppPreprocessorYuv(imageProcessor, imagePath, yolov7Inputs, imagePreProcessInfos, deviceId);
    } else {
        if (DeviceManager::IsAscend310P()) {
            Image decodeImage;
            APP_ERROR ret = imageProcessor.Decode(imagePath, decodeImage, ImageFormat::BGR_888);
            if (ret != APP_ERR_OK) {
                LogError << "ImageProcessor decode failed.";
                return OpenCVPreProcessor(imagePath, yolov7Inputs, imagePreProcessInfos, deviceId);
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
                RESIZER_TF_KEEP_ASPECT_RATIO, minScale);
            imagePreProcessInfos.push_back(imagePreProcessInfo);
        } else {
            return OpenCVPreProcessor(imagePath, yolov7Inputs, imagePreProcessInfos, deviceId);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR E2eInfer(std::map<std::string, std::string> pathMap, int32_t deviceId, bool isYuvInput)
{
    std::string imagePath = pathMap["imgPath"];
    APP_ERROR ret = CheckFileVaild(imagePath);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    vector<Tensor> yolov7Inputs;
    std::vector<ResizedImageInfo> resizedImageInfos;
    ret = DvppPreprocessor(imagePath, yolov7Inputs, resizedImageInfos, deviceId, isYuvInput);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    string modelPath = pathMap["modelPath"];
    ret = CheckFileVaild(modelPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    Model yolov7(modelPath, deviceId);

    vector<Tensor> yolov7Outputs = yolov7.Infer(yolov7Inputs);
    if (yolov7Outputs.size() == 0) {
        LogError << "YOLOv7 infer failed.";
        return APP_ERR_COMM_FAILURE;
    }
    for (size_t i = 0; i < yolov7Outputs.size(); i++) {
        yolov7Outputs[i].ToHost();
    }
    std::vector<Rect> cropConfigVec;
    string yolov7ConfigPath = pathMap["modelConfigPath"];
    string yolov7LabelPath = pathMap["modelLabelPath"];

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

void usage()
{
    std::cout << "Usage: " << std::endl;
    std::cout << "./sample -m model_path -c model_config_path -l model_label_path -i image_path [-y] " << std::endl;
}

int main(int argc, char *argv[])
{
    MxInit();
    if (argc > ARG_NUM || argc < ARG_NUM - 1) {
        usage();
        return 0;
    }
    int32_t deviceId = 0;
    bool isYuvInput = 0;
    std::map<std::string, std::string> pathMap;
    int input;
    const char* optString = "i:m:c:l:yh";
    while ((input = getopt(argc, argv, optString)) != -1) {
        switch (input) {
            case 'm':
                pathMap.insert({ "modelPath", optarg });
                break;
            case 'i':
                pathMap.insert({ "imgPath", optarg });
                break;
            case 'c':
                pathMap.insert({ "modelConfigPath", optarg });
                break;
            case 'l':
                pathMap.insert({ "modelLabelPath", optarg });
                break;
            case 'y':
                isYuvInput = true;
                break;
            case 'h':
                usage();
                return 0;
            case '?':
                usage();
                return 0;
        }
    }
    if (pathMap.count("modelPath") <= 0 || pathMap.count("imgPath") <= 0 || pathMap.count("modelConfigPath") <= 0 ||
        pathMap.count("modelLabelPath") <= 0) {
        LogError << "Invalid input params";
        usage();
        return 0;
    }
    APP_ERROR ret = E2eInfer(pathMap, deviceId, isYuvInput);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to run E2eInfer";
    }
    return 0;
}