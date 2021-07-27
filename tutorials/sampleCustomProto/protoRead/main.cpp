/*
 * Copyright (c) 2020.Huawei Technologies Co., Ltd. All rights reserved.
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

#include <cstring>
#include "MxBase/Log/Log.h"
#include "MxStream/StreamManager/MxStreamManager.h"
#include "mxpiSampleProto.pb.h"
namespace {
APP_ERROR ReadFile(const std::string& filePath, MxStream::MxstDataInput& dataBuffer)
{
    char c[PATH_MAX + 1] = { 0x00 };
    size_t count = filePath.copy(c, PATH_MAX + 1);
    if (count != filePath.length()) {
        LogError << "Failed to copy file path(" << c << ").";
        return APP_ERR_COMM_FAILURE;
    }
    // Get the absolute path of input file
    char path[PATH_MAX + 1] = { 0x00 };
    if ((strlen(c) > PATH_MAX) || (realpath(c, path) == nullptr)) {
        LogError << "Failed to get image, the image path is (" << filePath << ").";
        return APP_ERR_COMM_NO_EXIST;
    }
    // Open file with reading mode
    FILE *fp = fopen(path, "rb");
    if (fp == nullptr) {
        LogError << "Failed to open file (" << path << ").";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // Get the length of input file
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    // If file not empty, read it into FileInfo and return it
    if (fileSize > 0) {
        dataBuffer.dataSize = fileSize;
        dataBuffer.dataPtr = new (std::nothrow) uint32_t[fileSize];
        if (dataBuffer.dataPtr == nullptr) {
            LogError << "allocate memory with \"new uint32_t\" failed.";
            return APP_ERR_COMM_FAILURE;
        }

        uint32_t readRet = fread(dataBuffer.dataPtr, 1, fileSize, fp);
        if (readRet <= 0) {
            fclose(fp);
            return APP_ERR_COMM_READ_FAIL;
        }
        fclose(fp);
        return APP_ERR_OK;
    }
    fclose(fp);
    return APP_ERR_COMM_FAILURE;
}

std::string ReadPipelineConfig(const std::string& pipelineConfigPath)
{
    std::ifstream file(pipelineConfigPath.c_str(), std::ifstream::binary);
    if (!file) {
        LogError << pipelineConfigPath <<" file dose not exist.";
        return "";
    }
    file.seekg(0, std::ifstream::end);
    uint32_t fileSize = file.tellg();
    file.seekg(0);
    std::unique_ptr<char[]> data(new char[fileSize]);
    file.read(data.get(), fileSize);
    file.close();
    std::string pipelineConfig(data.get(), fileSize);
    return pipelineConfig;
}
}

int main(int argc, char* argv[])
{
    // read image file and build stream input
    MxStream::MxstDataInput dataBuffer;
    APP_ERROR ret = ReadFile("./test.jpg", dataBuffer);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Failed to read image file.";
        return ret;
    }
    // read pipeline config file
    std::string pipelineConfigPath = "../pipeline/Sample_proto.pipeline";
    std::string pipelineConfig = ReadPipelineConfig(pipelineConfigPath);
    if (pipelineConfig == "") {
        LogError << "Read pipeline failed.";
        return APP_ERR_COMM_INIT_FAIL;
    }
    // init stream manager
    MxStream::MxStreamManager mxStreamManager;
    ret = mxStreamManager.InitManager();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Failed to init Stream manager.";
        return ret;
    }
    // create stream by pipeline config file
    ret = mxStreamManager.CreateMultipleStreams(pipelineConfig);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Failed to create Stream.";
        return ret;
    }
    std::string streamName = "classification+detection";
    int inPluginId = 0;
    // send data into stream
    ret = mxStreamManager.SendData(streamName, inPluginId, dataBuffer);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Failed to send data to stream.";
        return ret;
    }

    // choose which metadata to be got.In this case we use the custom "mxpi_sampleproto"
    std::vector<std::string> keyVec;
    keyVec.push_back("mxpi_sampleproto");

    // get stream output
    std::vector<MxStream::MxstProtobufOut> output = mxStreamManager.GetProtobuf(streamName, inPluginId, keyVec);
    if (output.size() == 0) {
        LogError << "output size is 0";
        return APP_ERR_ACL_FAILURE;
    }
    if (output[0].errorCode != APP_ERR_OK) {
        LogError << "GetProtobuf error. errorCode=" << output[0].errorCode;
        return output[0].errorCode;
    }
    LogInfo << "key=" << output[0].messageName;
    LogInfo << "value=" << output[0].messagePtr.get()->DebugString();

    auto protoResList = std::static_pointer_cast<sampleproto::MxpiSampleProtoList>(output[0].messagePtr);
    LogInfo << "protobuf stringSample=" << protoResList->sampleprotovec(0).stringsample();

    // destroy streams
    mxStreamManager.DestroyAllStreams();
    delete dataBuffer.dataPtr;
    dataBuffer.dataPtr = nullptr;

    //delete output;
    return 0;
}