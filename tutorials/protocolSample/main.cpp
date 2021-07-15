/*
* Copyright (c) Huawei Technologies Co., Ltd. 2012-2021. All rights reserved.
* Description: protocol C++ sample
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

// protocol_example Does not contain running part in main()
namespace protocol_example {
    constexpr int COS_VALUE_INPUT = 0;

#pragma region proto_encode_example
    APP_ERROR MakeMxpiFrame(MxStream::MxstDataInput dataBuffer,
                            const std::string& elementName,
                            MxStream::MxstProtobufIn &protoBufBuffer)
    {
        // creat a MxpiVisionList object using shared_ptr
        auto visionList = std::make_shared<MxTools::MxpiVisionList>();

        // Appends a new MxpiVision element to the end of the MxpiVisionList field.
        MxTools::MxpiVision *mxpiVision = visionList->add_visionvec();

        // mutable_* creat a pointer to the mutable object,which mean you can change it`s value later
        MxTools::MxpiVisionInfo *visioninfo = mxpiVision->mutable_visioninfo();
        visioninfo->set_format(COS_VALUE_INPUT); // set_* change a exist object value
        visioninfo->set_width(COS_VALUE_INPUT);
        visioninfo->set_height(COS_VALUE_INPUT);
        visioninfo->set_widthaligned(COS_VALUE_INPUT);
        visioninfo->set_heightaligned(COS_VALUE_INPUT);

        MxTools::MxpiVisionData *visiondata = mxpiVision->mutable_visiondata();
        visiondata->set_dataptr((uint64_t) dataBuffer.dataPtr);
        visiondata->set_datasize(dataBuffer.dataSize);
        visiondata->set_memtype(MxTools::MXPI_MEMORY_HOST_NEW);

        // MxpiFrameInfo
        auto frameInfo = std::make_shared<MxTools::MxpiFrameInfo>();
        frameInfo->set_channelid(0);
        frameInfo->set_frameid(0);

        // MxpiFrame
        auto frameBuffer = std::make_shared<MxTools::MxpiFrame>();
        frameBuffer->mutable_frameinfo()->CopyFrom(*frameInfo); // copy data from share_ptr to struct
        frameBuffer->mutable_visionlist()->CopyFrom(*visionList);

        // write single protobuf
        protoBufBuffer.key = elementName;
        protoBufBuffer.messagePtr = std::static_pointer_cast<google::protobuf::Message>(frameBuffer);

        return APP_ERR_OK;
    }

    APP_ERROR MakeMxpiVisionList(MxStream::MxstDataInput dataBuffer,
                                 const std::string& elementName,
                                 MxStream::MxstProtobufIn &protoBufBuffer)
    {
        // MxpiVisionList
        auto visionList = std::make_shared<MxTools::MxpiVisionList>();

        MxTools::MxpiVision *mxpiVision = visionList->add_visionvec();

        MxTools::MxpiVisionInfo *visioninfo = mxpiVision->mutable_visioninfo();
        visioninfo->set_format(COS_VALUE_INPUT);
        visioninfo->set_width(COS_VALUE_INPUT);
        visioninfo->set_height(COS_VALUE_INPUT);
        visioninfo->set_widthaligned(COS_VALUE_INPUT);
        visioninfo->set_heightaligned(COS_VALUE_INPUT);

        MxTools::MxpiVisionData *visiondata = mxpiVision->mutable_visiondata();
        visiondata->set_dataptr((uint64_t) dataBuffer.dataPtr);
        visiondata->set_datasize(dataBuffer.dataSize);
        visiondata->set_memtype(MxTools::MXPI_MEMORY_HOST_NEW);

        // write single protobuf
        protoBufBuffer.key = elementName;
        protoBufBuffer.messagePtr = std::static_pointer_cast<google::protobuf::Message>(visionList);

        return APP_ERR_OK;
    }

#pragma endregion proto_encode_example

#pragma region proto_decode_example
    APP_ERROR ReadMxpiVisionList(MxTools::MxpiVisionList visionList)
    {
        // MxpiVisionList
        auto formatInt = visionList.visionvec(0).visioninfo().format();
        LogInfo << "From visionList/visionvec[0]/visioninfo/format decode int:" << formatInt;
        return APP_ERR_OK;
    }
#pragma endregion proto_decode_example
}

// Actual running part of main()
namespace {
    constexpr int COS_LEN_VALUE = 416;
    constexpr int COS_FORMAT_VALUE = 12;
    constexpr bool USE_SENDDATA = false; // switch different send data method

    // read data File to MxstDataInput structure
    APP_ERROR ReadFile(const std::string& filePath, MxStream::MxstDataInput& dataBuffer)
    {
        char filePathChar[PATH_MAX + 1] = {0x00 };
        size_t count = filePath.copy(filePathChar, PATH_MAX + 1);
        if (count != filePath.length()) {
            LogError << "Failed to copy file absPath(" << filePathChar << ").";
            return APP_ERR_COMM_FAILURE;
        }
        // Get the absolute Path of input file
        char absPath[PATH_MAX + 1] = {0x00 };
        if ((strlen(filePathChar) > PATH_MAX) || (realpath(filePathChar, absPath) == nullptr)) {
            LogError << "Failed to get image, the image absPath is (" << filePath << ").";
            return APP_ERR_COMM_NO_EXIST;
        }
        // Open file with reading mode
        FILE *fp = fopen(absPath, "rb");
        if (fp == nullptr) {
            LogError << "Failed to open file (" << absPath << ").";
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
                fclose(fp);
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

    // read pipeline config
    std::string ReadPipelineConfig(const std::string& pipelineConfigPath)
    {
        std::ifstream file(pipelineConfigPath.c_str(), std::ifstream::binary);
        if (file.is_open() == false) {
            LogError << pipelineConfigPath << " file dose not exist.";
            return "";
        }
        file.seekg(0, std::ifstream::end);
        uint32_t fileSize = file.tellg();
        file.seekg(0);
        auto data = std::unique_ptr<char>(new char[fileSize]);
        file.read(data.get(), fileSize);
        file.close();
        std::string pipelineConfig(data.get(), fileSize);
        return pipelineConfig;
    }

    // make proto data for SendData()
    APP_ERROR MakeSendDataProto(MxStream::MxstDataInput dataBuffer, const std::string& elementName,
                                std::vector<MxStream::MxstMetadataInput> &metedataVec,
                                MxStream::MxstBufferInput &bufferInput)
    {
        // MakeMxpiFrame
        auto visionList = std::make_shared<MxTools::MxpiVisionList>();
        MxTools::MxpiVision *mxpiVision = visionList->add_visionvec();
        MxTools::MxpiVisionInfo *visioninfo = mxpiVision->mutable_visioninfo();
        // lack param information!!!! These value is just for example!(in this case decided by model)
        visioninfo->set_format(COS_FORMAT_VALUE);
        visioninfo->set_width(COS_LEN_VALUE);
        visioninfo->set_height(COS_LEN_VALUE);
        visioninfo->set_widthaligned(COS_LEN_VALUE);
        visioninfo->set_heightaligned(COS_LEN_VALUE);

        MxTools::MxpiVisionData *visiondata = mxpiVision->mutable_visiondata();
        visiondata->set_dataptr((uint64_t) dataBuffer.dataPtr);
        visiondata->set_datasize(dataBuffer.dataSize);
        visiondata->set_memtype(MxTools::MXPI_MEMORY_HOST_NEW);

        auto frameInfo = std::make_shared<MxTools::MxpiFrameInfo>();
        frameInfo->set_channelid(0);
        frameInfo->set_frameid(0);

        MxStream::MxstMetadataInput metedataInput;
        metedataInput.dataSource = elementName;
        metedataInput.messagePtr = std::static_pointer_cast<google::protobuf::Message>(visionList);

        auto frameBuffer = std::make_shared<MxTools::MxpiFrame>();
        frameBuffer->mutable_frameinfo()->CopyFrom(*frameInfo);

        // write proto data structure
        bufferInput.mxpiFrameInfo = *frameInfo;
        bufferInput.mxpiVisionInfo = *visioninfo;
        bufferInput.dataSize = dataBuffer.dataSize;
        bufferInput.dataPtr = dataBuffer.dataPtr;

        metedataVec.push_back(metedataInput);

        return APP_ERR_OK;
    }

    // make proto data for SendProtobuf()
    APP_ERROR MakeSendProtobuf(MxStream::MxstDataInput dataBuffer, const std::string& elementName,
                                MxStream::MxstProtobufIn &protoBufBuffer)
    {
        // MxpiVisionList
        auto visionList = std::make_shared<MxTools::MxpiVisionList>();

        MxTools::MxpiVision *mxpiVision = visionList->add_visionvec();

        MxTools::MxpiVisionInfo *visioninfo = mxpiVision->mutable_visioninfo();
        visioninfo->set_format(COS_FORMAT_VALUE);
        visioninfo->set_width(COS_LEN_VALUE);
        visioninfo->set_height(COS_LEN_VALUE);
        visioninfo->set_widthaligned(COS_LEN_VALUE);
        visioninfo->set_heightaligned(COS_LEN_VALUE);

        MxTools::MxpiVisionData *visiondata = mxpiVision->mutable_visiondata();
        visiondata->set_dataptr((uint64_t) dataBuffer.dataPtr);
        visiondata->set_datasize(dataBuffer.dataSize);
        visiondata->set_memtype(MxTools::MXPI_MEMORY_HOST_NEW);

        protoBufBuffer.key = elementName;
        protoBufBuffer.messagePtr = std::static_pointer_cast<google::protobuf::Message>(visionList);
        return APP_ERR_OK;
    }

    // send data to stream and get result
    APP_ERROR SendData2Stream(MxStream::MxStreamManager& mxStreamManager, MxStream::MxstDataInput& dataBuffer,
                                const std::string streamName, const int& inPluginId, const std::string& elementName)
    {
        APP_ERROR ret = APP_ERR_OK;
        // build proto data with SendData()
        MxStream::MxstBufferInput bufferInput;
        std::vector<MxStream::MxstMetadataInput> metedataVec;

        ret = MakeSendDataProto(dataBuffer, elementName, metedataVec, bufferInput);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Failed to MakeProto data.";
            return ret;
        }

        // send data into stream
        ret = mxStreamManager.SendData(streamName, elementName, metedataVec, bufferInput);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Failed to send data into stream.";
            return ret;
        }
        // get stream output
        MxStream::MxstDataOutput* output = mxStreamManager.GetResult(streamName, inPluginId);
        if (output == nullptr) {
            LogError << "Failed to get pipeline output.";
            return ret;
        }

        std::string result = std::string((char *)output->dataPtr, output->dataSize);
        LogWarn << "Output:" << result;
        delete output;
        output = nullptr;
        return APP_ERR_OK;
    }

    // send protobuf to stream and get result
    APP_ERROR SendProto2Stream(MxStream::MxStreamManager& mxStreamManager, MxStream::MxstDataInput& dataBuffer,
                                const std::string streamName, const int& inPluginId, const std::string& elementName)
    {
        APP_ERROR ret = APP_ERR_OK;
        // build proto data with SendProtobuf()
        MxStream::MxstProtobufIn protoBufBuffer;
        ret = MakeSendProtobuf(dataBuffer, elementName, protoBufBuffer);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Failed to MakeProto data.";
            return ret;
        }
        std::vector<MxStream::MxstProtobufIn> dataBufferVec;
        dataBufferVec.push_back(protoBufBuffer);

        ret = mxStreamManager.SendProtobuf(streamName, inPluginId, dataBufferVec);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Failed to send data into stream.";
            return ret;
        }

        std::vector<std::string> keyVec;
        keyVec.push_back(protoBufBuffer.key);
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
        output.clear();

        return APP_ERR_OK;
    }
}

/* dataFile:./test.jpg
 * pipelineFile:./pipeSample.pipeline
 * Func:display metadata structure in stream
 * reference:MxpiDataType.proto
 * */
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
    std::string pipelineConfig = ReadPipelineConfig("./pipeSample.pipeline");
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

    // send data to stream and get result
    std::string streamName = "pipeSample";
    int inPluginId = 0;
    std::string elementName = "appsrc0";

    if (USE_SENDDATA == true) {
        ret = SendData2Stream(mxStreamManager, dataBuffer, streamName, inPluginId, elementName);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Failed process SendData() to Stream and get result.";
            return ret;
        }
    } else {
        ret = SendProto2Stream(mxStreamManager, dataBuffer, streamName, inPluginId, elementName);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Failed process SendProtobuf() to Stream and get result.";
            return ret;
        }
    }

    // destroy streams
    mxStreamManager.DestroyAllStreams();

    delete dataBuffer.dataPtr;
    dataBuffer.dataPtr = nullptr;

    return 0;
}