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

#include <dirent.h>
#include <cstring>
#include <unistd.h>
#include <thread>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include "MxBase/Log/Log.h"
#include "MxStream/StreamManager/MxStreamManager.h"
#include "MxBase/DeviceManager/DeviceManager.h"

using namespace MxTools;
using namespace MxStream;
using namespace cv;

namespace {
    const int TIME_OUT = 15000;
    const int INPUT_UINT8 = 1;
}

std::string ReadFileContent(const std::string filePath)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        LogError << "Invalid file. filePath(" << filePath << ")";
        return "";
    }

    file.seekg(0, std::ifstream::end);
    uint32_t fileSize = file.tellg();
    file.seekg(0);
    std::vector<char> buffer = {};
    buffer.resize(fileSize);
    file.read(buffer.data(), fileSize);
    file.close();

    return std::string(buffer.data(), fileSize);
}

static void GetDataBuf(std::vector<MxStream::MxstProtobufIn>& dataBufferVec, MxStream::MxstDataInput& dataInput,
    int width, int height)
{
    std::shared_ptr<MxTools::MxpiVisionList> objectList = std::make_shared<MxTools::MxpiVisionList>();
    MxTools::MxpiVision* mxpiVision = objectList->add_visionvec();
    MxTools::MxpiVisionInfo *visionInfo =  mxpiVision->mutable_visioninfo();
    const int format = 12;
    visionInfo->set_format(format);
    visionInfo->set_width(width);
    visionInfo->set_height(height);
    visionInfo->set_heightaligned(height);
    visionInfo->set_widthaligned(width);

    MxTools::MxpiVisionData *visionData =  mxpiVision->mutable_visiondata();
    visionData->set_dataptr((uint64_t)dataInput.dataPtr);
    visionData->set_datasize(dataInput.dataSize);
    visionData->set_memtype(MxTools::MXPI_MEMORY_HOST_NEW);
    MxStream::MxstProtobufIn dataBuffer;
    dataBuffer.key = "appsrc1";
    dataBuffer.messagePtr = std::static_pointer_cast<google::protobuf::Message>(objectList);
    dataBufferVec.push_back(dataBuffer);
}

APP_ERROR GetOpenCVDataBuf(std::vector<MxStream::MxstProtobufIn>& dataBufferVec, int dataType, std::string filePath,
    int modelWidth, int modelHeight)
{
    if (dataType <= 0) {
        LogError << "The datatype must be larger than 0";
        return APP_ERR_COMM_FAILURE;
    }
    char c[PATH_MAX + 1] = { 0x00 };
    size_t count = filePath.copy(c, PATH_MAX + 1);
    char realPath[PATH_MAX + 1] = { 0x00 };
    if (count != filePath.length() || (realpath(c, realPath) == nullptr)) {
        LogError << "Failed to get image, the image path is (" << filePath << ").";
        return APP_ERR_COMM_NO_EXIST;
    }
    Mat img = imread(realPath, 1);
    if (img.empty()) {
        LogError << "Can not read this picture(" << realPath << ")";
        return APP_ERR_COMM_NO_EXIST;
    }
    int height = img.rows;
    int width = img.cols;
    Mat shrink;
    if (height >= modelHeight && width >= modelWidth) {
        Size dsize = Size(round(modelWidth), round(modelHeight));
        resize(img, shrink, dsize, 0, 0, INTER_AREA);
    } else {
        float fx = modelWidth / (float)width;
        float fy = modelHeight / (float)height;
        Mat enlarge;
        resize(img, shrink, Size(), fx, fy, INTER_CUBIC);
    }
    height = shrink.rows;
    width = shrink.cols;
    Mat yuvImg = {};
    MxStream::MxstDataInput dataInput;
    const int convert_3 = 3;
    const int convert_2 = 2;
    dataInput.dataSize = width * height * dataType * convert_3 / convert_2;
    cvtColor(shrink, yuvImg, COLOR_RGB2YUV_I420);
    dataInput.dataPtr = new (std::nothrow) uint32_t[dataInput.dataSize];
    std::copy(yuvImg.data, yuvImg.data + dataInput.dataSize / dataType, (char*)dataInput.dataPtr);
    GetDataBuf(dataBufferVec, dataInput, modelWidth, modelHeight);
    LogDebug << "width: " << width << ", height: " << height << ", dataSize: " << dataInput.dataSize;
    return APP_ERR_OK;
}

APP_ERROR GetPicture(std::string filePath, std::vector<std::string>& pictureName)
{
    pictureName.clear();
    std::string filename;
    DIR *pDir = nullptr;
    struct dirent *ptr = nullptr;
    if (!(pDir = opendir(filePath.c_str()))) {
        LogError << "Folder doesn't Exist!";
        return APP_ERR_COMM_NO_EXIST;
    }
    int strLen = 4;
    while ((ptr = readdir(pDir)) != nullptr) {
        std::string tmpStr = ptr->d_name;
        if (tmpStr.length() < strLen) {
            continue;
        }
        tmpStr = tmpStr.substr(tmpStr.length() - strLen, tmpStr.length());
        if (tmpStr == ".jpg" || tmpStr == ".JPG") {
            filename = filePath + "/" + ptr->d_name;
            pictureName.push_back(filename);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR streamCallback(MxStreamManager& mxStreamManager, std::string streamName, std::string picturePath)
{
    std::vector<std::string> pictureName = {};
    auto ret = GetPicture(picturePath, pictureName);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to get picture";
        return ret;
    }
    for (int i = 0; i < pictureName.size(); ++i) {
        LogInfo << "Start to send picture(" << pictureName[i] << ").";
        MxstDataInput mxstDataInput = {};
        std::string catImage = ReadFileContent(pictureName[i]);
        mxstDataInput.dataPtr = (uint32_t *) catImage.c_str();
        mxstDataInput.dataSize = catImage.size();
        ret = mxStreamManager.SendData(streamName, 0, mxstDataInput);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to send data to stream";
            continue;
        }
        MxstDataOutput *outputPtr = mxStreamManager.GetResult(streamName, 0, TIME_OUT);
        if (outputPtr == nullptr || outputPtr->errorCode != 0) {
            LogError << "Failed to get data to stream";
            continue;
        }
        std::string dataStr = std::string((char *)outputPtr->dataPtr, outputPtr->dataSize);
        LogInfo << "[" << streamName << "] GetResult: " << dataStr;
    }
    return APP_ERR_OK;
}

APP_ERROR TestMultiThread(std::string pipelinePath)
{
    LogInfo << "********case TestMultiThread********" << std::endl;
    MxStream::MxStreamManager mxStreamManager;
    APP_ERROR ret = mxStreamManager.InitManager();
    if (ret != APP_ERR_OK) {
        LogError << "Failed to init streammanager";
        return ret;
    }
    ret = mxStreamManager.CreateMultipleStreamsFromFile(pipelinePath);
    if (ret != APP_ERR_OK) {
        LogError << "Pipeline is no exit";
        return ret;
    }

    int threadCount = 3;
    std::thread threadSendData[threadCount];
    std::string streamName[threadCount];
    std::string picturePath = "../picture";
    for (int i = 0; i < threadCount; ++i) {
        streamName[i] = "detection" + std::to_string(i);
        threadSendData[i] = std::thread(streamCallback, std::ref(mxStreamManager), streamName[i], picturePath);
    }
    for (int j = 0; j < threadCount; ++j) {
        threadSendData[j].join();
    }

    ret = mxStreamManager.DestroyAllStreams();
    if (ret != APP_ERR_OK) {
        LogError << "Failed to destroy stream";
    }
    return APP_ERR_OK;
}

APP_ERROR sendDataCallback(MxStreamManager& mxStreamManager, std::string streamName,
    std::vector<std::string>& pictureName, int width, int height)
{
    APP_ERROR ret;
    MxBase::DeviceManager* m = MxBase::DeviceManager::GetInstance();
    MxBase::DeviceContext deviceContext;
    deviceContext.devId = 0;
    m->InitDevices();
    m->SetDevice(deviceContext);
    for (int i = 0; i < pictureName.size(); ++i) {
        std::vector<MxStream::MxstProtobufIn> dataBufferVec;
        ret = GetOpenCVDataBuf(dataBufferVec, INPUT_UINT8, pictureName[i], width, height);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to get data buf";
            m->DestroyDevices();
            return ret;
        }
        LogInfo << "Start to send picture(" << pictureName[i] << ").";
        ret = mxStreamManager.SendProtobuf(streamName, 0, dataBufferVec);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to send protobuf";
            m->DestroyDevices();
            return ret;
        }
    }
    m->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR getDataCallback(MxStreamManager& mxStreamManager, std::string streamName, int count)
{
    std::vector<std::string> strvec = {};
    strvec.push_back("mxpi_modelinfer0");
    for (int i = 0; i < count; ++i) {
        std::vector<MxstProtobufOut> bufvec = mxStreamManager.GetProtobuf(streamName, 0, strvec);
        if (bufvec[0].errorCode != APP_ERR_OK) {
            LogError << "Failed to get protobuf";
            continue;
        }
        for (int j = 0; j < bufvec.size(); ++j) {
            LogInfo << "Value(" << streamName << ") = " << bufvec[0].messagePtr.get()->DebugString();
        }
    }
    return APP_ERR_OK;
}

APP_ERROR TestSendProtobuf(std::string pipelinePath)
{
    LogInfo << "********case TestSendProtobuf********";
    MxStreamManager mxStreamManager;
    APP_ERROR ret = mxStreamManager.InitManager();
    if (ret != APP_ERR_OK) {
        LogError << "Failed to init streammanager";
        return ret;
    }
    ret = mxStreamManager.CreateMultipleStreamsFromFile(pipelinePath);
    if (ret != APP_ERR_OK) {
        LogError << "Pipeline is no exit";
        return ret;
    }
    std::vector<std::string> pictureName = {};
    std::string picturePath = "../picture";
    ret = GetPicture(picturePath, pictureName);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to get picture";
        return ret;
    }
    const int threadCount = 4;
    std::thread threadSendData[threadCount];
    std::thread threadGetData[threadCount];
    std::string streamName[threadCount];
    int width[threadCount] = {416, 416, 416, 416};
    int height[threadCount] = {416, 416, 416, 416};
    for (int i = 0; i < threadCount; ++i) {
        streamName[i] = "detection" + std::to_string(i);
        threadGetData[i] = std::thread(getDataCallback, std::ref(mxStreamManager), streamName[i], pictureName.size());
        threadSendData[i] = std::thread(sendDataCallback, std::ref(mxStreamManager), streamName[i],
            std::ref(pictureName), width[i], height[i]);
    }

    for (int j = 0; j < threadCount; ++j) {
        threadSendData[j].join();
        threadGetData[j].join();
    }

    ret = mxStreamManager.DestroyAllStreams();
    if (ret != APP_ERR_OK) {
        LogError << "Failed to destroy stream";
    }
    return APP_ERR_OK;
}

int main(int argc, char *argv[])
{
    if (argc == 1) {
        LogWarn << "Parameter cannot be empty";
        return 0;
    }
    std::string type = argv[1];
    struct timeval inferStartTime = { 0 };
    struct timeval inferEndTime = { 0 };
    gettimeofday(&inferStartTime, nullptr);
    APP_ERROR ret;
    if (type == "0") {
        ret = TestMultiThread("EasyStream.pipeline");
    } else {
        ret = TestSendProtobuf("EasyStream_protobuf.pipeline");
    }
    if (ret == APP_ERR_OK) {
        float SEC2MS = 1000.0;
        gettimeofday(&inferEndTime, nullptr);
        double inferCostTime = SEC2MS * (inferEndTime.tv_sec - inferStartTime.tv_sec) +
                               (inferEndTime.tv_usec - inferStartTime.tv_usec) / SEC2MS;
        LogInfo << "Total time: " << inferCostTime / SEC2MS;
    }
    return 0;
}
