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

#include <dirent.h>
#include <cstring>
#include <unistd.h>
#include <thread>
#include "MxBase/Log/Log.h"
#include "MxStream/StreamManager/MxStreamManager.h"

using namespace MxTools;
using namespace MxStream;

namespace {
    const int TIME_OUT = 5000;
    const float SEC2MS = 1000.0;
    const std::string PICTURE_PATH = "../input_data/";
}

std::string ReadFileContent(const std::string& filePath)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        LogError << "Invalid file, filePath(" << filePath << ")";
        return "";
    }

    file.seekg(0, std::ifstream::end);
    uint32_t fileSize = file.tellg();
    file.seekg(0);
    std::vector<char> buffer(fileSize);
    file.read(buffer.data(), fileSize);
    file.close();

    return std::string(buffer.data(), fileSize);
}

APP_ERROR GetPicture(const std::string& filePath, std::vector<std::string>& pictureName)
{
    pictureName.clear();
    std::string filename;
    DIR *pDir = nullptr;
    struct dirent *ptr = nullptr;
    if (!(pDir = opendir(filePath.c_str()))) {
        LogError << "Folder doesn't Exist!";
        return APP_ERR_COMM_NO_EXIST;
    }
    constexpr int suffixLen = 4;
    while ((ptr = readdir(pDir)) != nullptr) {
        std::string tmpStr = ptr->d_name;
        if (tmpStr.length() < suffixLen) {
            continue;
        }
        tmpStr = tmpStr.substr(tmpStr.length() - suffixLen, tmpStr.length());
        if (tmpStr == ".jpg" || tmpStr == ".JPG") {
            filename = filePath + "/" + ptr->d_name;
            pictureName.push_back(filename);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR GetCallback(MxStreamManager& mxStreamManager, const std::string& streamName, int count)
{
    for (int i = 0; i < count; ++i) {
        MxStream::MxstDataOutput* output = mxStreamManager.GetResult(streamName, 0, TIME_OUT);
        if (output == nullptr) {
            LogError << "Failed to get pipeline output.";
            return APP_ERR_COMM_FAILURE;
        }
        std::string dataStr = std::string((char *)output->dataPtr, output->dataSize);
        LogInfo << "[" << streamName << "] GetResult: " << dataStr;
    }
    return APP_ERR_OK;
}

APP_ERROR SendCallback(MxStreamManager& mxStreamManager,
    const std::string& streamName, std::vector<std::string>& pictureName)
{
    APP_ERROR ret = APP_ERR_OK;
    for (unsigned int i = 0; i < pictureName.size(); ++i) {
        // send data into stream
        MxstDataInput mxstDataInput = {};
        std::string catImage = ReadFileContent(pictureName[i]);
        mxstDataInput.dataPtr = (uint32_t *) catImage.c_str();
        mxstDataInput.dataSize = catImage.size();

        ret = mxStreamManager.SendData(streamName, 0, mxstDataInput);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Failed to send data to stream.";
            return ret;
        }
    }
    return APP_ERR_OK;
}
APP_ERROR TestMultiThread(const std::string& pipelinePath)
{
    LogDebug << "********case TestMultiThread********" << std::endl;

    MxStream::MxStreamManager mxStreamManager;
    APP_ERROR ret = mxStreamManager.InitManager();
    if (ret != APP_ERR_OK) {
        LogError << "Failed to init streammanager";
        return ret;
    }
    ret = mxStreamManager.CreateMultipleStreamsFromFile(pipelinePath);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Failed to create Stream.";
        return ret;
    }

    std::vector<std::string> pictureName;
    ret = GetPicture(PICTURE_PATH, pictureName);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to get picture";
        return ret;
    }

    int threadCount = 3;
    std::thread threadSendData[threadCount];
    std::thread threadGetData[threadCount];
    std::string streamName[threadCount];

    for (int i = 0; i < threadCount; ++i) {
        streamName[i] = "OCR" + std::to_string(i);
        threadGetData[i] = std::thread(GetCallback, std::ref(mxStreamManager), streamName[i], pictureName.size());
        threadSendData[i] = std::thread(SendCallback, std::ref(mxStreamManager), streamName[i], std::ref(pictureName));
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
    struct timeval inferStartTime = { 0 };
    struct timeval inferEndTime = { 0 };
    gettimeofday(&inferStartTime, nullptr);
    // read pipeline config file
    std::string pipelineConfigPath = "../data/OCR_multi3.pipeline";

    APP_ERROR ret = APP_ERR_OK;
    ret = TestMultiThread(pipelineConfigPath);
    if (ret == APP_ERR_OK) {
        gettimeofday(&inferEndTime, nullptr);
        double inferCostTime = SEC2MS * (inferEndTime.tv_sec - inferStartTime.tv_sec) +
                               (inferEndTime.tv_usec - inferStartTime.tv_usec) / SEC2MS;
        LogInfo << "Total time: " << inferCostTime / SEC2MS;
    }
    return 0;
}
