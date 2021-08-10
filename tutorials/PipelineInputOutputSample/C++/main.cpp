/*
 * Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.
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
 
#include <string>
#include "MxBase/Log/Log.h"
#include "MxStream/StreamManager/MxStreamManager.h"

enum OperationMode {
    SENDDATA_GETRESULT = 0,
    SENDDATA_GETRESULT_WITH_UNIQUE_ID,
    SENDPROTOBUFFER_GETPROBUFFER

};

// SendData-GetResult 第1种模式
static APP_ERROR SendDataAndGetResult(std::shared_ptr<MxStream::MxStreamManager> mxStreamManager,
                                      const std::string &streamName, const uint32_t &inPluginId,
                                      const uint32_t &outPluginId) {
    if (mxStreamManager.get() == nullptr) {
        LogError << "mxStreamManager is nullptr.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    // 构建输入数据
    MxStream::MxstDataInput mxstDataInput = {};
    std::string buffer = "hello";
    mxstDataInput.dataSize = buffer.size();
    mxstDataInput.dataPtr = (uint32_t *) &buffer[0];
    APP_ERROR ret = mxStreamManager->SendData(streamName, inPluginId, mxstDataInput);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Send failed.";
        return ret;
    }
    // 获取输出数据
    MxStream::MxstDataOutput *mxstDataOutput = mxStreamManager->GetResult(streamName, outPluginId);
    if (mxstDataOutput == nullptr) {
        LogError << "mxstDataOutput is nullptr.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    std::string result((char *) mxstDataOutput->dataPtr, mxstDataOutput->dataSize);
    LogInfo << "results:" << result;
    return APP_ERR_OK;
}

// SendData-GetResult 第2种模式
static APP_ERROR SendDataAndGetResult(std::shared_ptr<MxStream::MxStreamManager> mxStreamManager,
                                      const std::string &streamName, const std::string &inElementName,
                                      const uint32_t &outPluginId) {
    if (mxStreamManager.get() == nullptr) {
        LogError << "SendDataAndGetResult mxStreamManager is nullptr.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    // 构建输入数据
    MxStream::MxstDataInput mxstDataInput = {};
    std::string buffer = "hello";
    mxstDataInput.dataSize = buffer.size();
    mxstDataInput.dataPtr = (uint32_t *) &buffer[0];
    // 传入具体的输入插件名称 inElementName
    APP_ERROR ret = mxStreamManager->SendData(streamName, inElementName, mxstDataInput);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "SendData two failed.";
        return ret;
    }
    // 获取输出数据
    MxStream::MxstDataOutput *mxstDataOutput = mxStreamManager->GetResult(streamName, outPluginId);
    if (mxstDataOutput == nullptr) {
        LogError << "SendDataAndGetResult mxstDataOutput is nullptr.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    std::string result((char *) mxstDataOutput->dataPtr, mxstDataOutput->dataSize);
    LogInfo << "results:" << result;
    return APP_ERR_OK;
}

// SendData-GetResult 第3种模式
static APP_ERROR SendDataAndGetResult(std::shared_ptr<MxStream::MxStreamManager> mxStreamManager,
                                      const std::string &streamName,
                                      const std::string &inElementName) {
    // Stream接收的数据结构体，存储的图像数据
    MxStream::MxstBufferInput mxstBufferInput;
    std::string buffer = "hello";
    mxstBufferInput.dataSize = buffer.size();
    mxstBufferInput.dataPtr = (uint32_t *) &buffer[0];

    std::shared_ptr<MxTools::MxpiTextsInfo> mxpiTextsInfo = std::make_unique<MxTools::MxpiTextsInfo>();
    mxpiTextsInfo->add_text("hello");
    // Stream接收的元数据结构体
    std::vector<MxStream::MxstMetadataInput> mxstMetadataInputVec;
    MxStream::MxstMetadataInput mxstMetadataInput;
    mxstMetadataInput.messagePtr = mxpiTextsInfo;
    mxstMetadataInput.dataSource = "appsrc0";
    mxstMetadataInputVec.push_back(mxstMetadataInput);

    APP_ERROR ret = mxStreamManager->SendData(streamName, "appsrc0", mxstMetadataInputVec, mxstBufferInput);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "SendData failed.";
        return ret;
    }

    // 想要获取数据的插件
    std::vector<std::string> dataSourceVec = {"appsrc0"};
    MxStream::MxstBufferAndMetadataOutput mxstBufferAndMetadataOutput =
            mxStreamManager->GetResult(streamName, "appsink0", dataSourceVec);
    if (mxstBufferAndMetadataOutput.errorCode != APP_ERR_OK) {
        LogError << GetError(mxstBufferAndMetadataOutput.errorCode) << "GetResult failed.";
        return mxstBufferAndMetadataOutput.errorCode;
    }
    if (mxstBufferAndMetadataOutput.bufferOutput.get() == nullptr) {
        LogError << "bufferOutput nullptr.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    // 将buffer拷贝到字符串中存储
    std::string outBuffer((char *) mxstBufferAndMetadataOutput.bufferOutput->dataPtr,
                          mxstBufferAndMetadataOutput.bufferOutput->dataSize);
    LogInfo << "results:" << outBuffer;

    // 打印元数据
    for (uint32_t i = 0; i < mxstBufferAndMetadataOutput.metadataVec.size(); i++) {
        auto metaData = mxstBufferAndMetadataOutput.metadataVec[i];
        LogInfo << metaData.dataPtr->DebugString();
    }
    return APP_ERR_OK;
}


// SendDataWithUniqueId-GetResultWithUniqueId 第1种模式
static APP_ERROR SendDataWithUniqueIdAndGetResultWithUniqueId(
        std::shared_ptr<MxStream::MxStreamManager> mxStreamManager,
        const std::string &streamName, const std::uint32_t &inPluginId) {
    if (mxStreamManager.get() == nullptr) {
        LogError << "mxStreamManager is nullptr.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    // 构建输入数据
    MxStream::MxstDataInput mxstDataInput = {};
    std::string buffer = "hello";
    mxstDataInput.dataSize = buffer.size();
    mxstDataInput.dataPtr = (uint32_t *) &buffer[0];
    uint64_t uniqueId = 0;
    APP_ERROR ret = mxStreamManager->SendDataWithUniqueId(streamName, inPluginId, mxstDataInput, uniqueId);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "SendData failed.";
        return ret;
    }
    const uint32_t waitTime = 1000;
    // get stream output
    auto *mxstDataOutput = mxStreamManager->GetResultWithUniqueId(streamName, uniqueId, waitTime);
    if (mxstDataOutput == nullptr) {
        LogError << "mxstDataOutput is nullptr.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    std::string result((char *) mxstDataOutput->dataPtr, mxstDataOutput->dataSize);
    LogInfo << "results:" << result;
    return APP_ERR_OK;
}

// SendDataWithUniqueId-GetResultWithUniqueId 第2种模式
static APP_ERROR SendDataWithUniqueIdAndGetResultWithUniqueId(
        std::shared_ptr<MxStream::MxStreamManager> mxStreamManager,
        const std::string &streamName, const std::string &elementName) {
    if (mxStreamManager.get() == nullptr) {
        LogError << "SendDataWithUniqueIdAndGetResultWithUniqueId mxStreamManager is nullptr.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    // 构建输入数据
    MxStream::MxstDataInput mxstDataInput = {};
    std::string buffer = "hello";
    mxstDataInput.dataSize = buffer.size();
    mxstDataInput.dataPtr = (uint32_t *) &buffer[0];
    uint64_t uniqueId = 0;
    APP_ERROR ret = mxStreamManager->SendDataWithUniqueId(streamName, elementName, mxstDataInput, uniqueId);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "SendData failed.";
        return ret;
    }
    const uint32_t waitTime = 1000;
    // get stream output
    auto *mxstDataOutput = mxStreamManager->GetResultWithUniqueId(streamName, uniqueId, waitTime);
    if (mxstDataOutput == nullptr) {
        LogError << "SendDataWithUniqueIdAndGetResultWithUniqueId mxstDataOutput is nullptr.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    std::string result((char *) mxstDataOutput->dataPtr, mxstDataOutput->dataSize);
    LogInfo << "results:" << result;
    return APP_ERR_OK;
}

// sendProtobuffer-GetProtobuffer
static APP_ERROR sendProtobufferAndGetProtobuffer(std::shared_ptr<MxStream::MxStreamManager> mxStreamManager,
                                                  const std::string &streamName, const std::uint32_t &inPluginId,
                                                  const uint32_t &outPluginId) {
    std::shared_ptr<MxTools::MxpiTextsInfo> mxpiTextsInfo = std::make_unique<MxTools::MxpiTextsInfo>();
    mxpiTextsInfo->add_text("hello");

    std::vector<MxStream::MxstProtobufIn> mxstProtobufInVec;
    MxStream::MxstProtobufIn mxstProtobufIn;
    mxstProtobufIn.messagePtr = mxpiTextsInfo;
    mxstProtobufIn.key = "appsrc0";
    mxstProtobufInVec.push_back(mxstProtobufIn);

    APP_ERROR ret = mxStreamManager->SendProtobuf(streamName, "appsrc0", mxstProtobufInVec);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "SendProtobuf failed.";
        return ret;
    }

    std::vector<std::string> keys = {"appsrc0"};
    std::vector<MxStream::MxstProtobufOut> mxstProtobufOutVec = mxStreamManager->GetProtobuf(streamName, outPluginId,
                                                                                             keys);

    for (uint32_t i = 0; i < mxstProtobufOutVec.size(); i++) {
        auto mxstProtobufOut = mxstProtobufOutVec[i];
        if (mxstProtobufOut.errorCode != APP_ERR_OK) {
            LogInfo << GetError(mxstProtobufOut.errorCode) << "GetProtobuf failed.";
            return mxstProtobufOut.errorCode;
        }
        LogInfo << "result:" << mxstProtobufOut.messagePtr->DebugString();
    }
    return APP_ERR_OK;
}


static APP_ERROR DetSendData(std::shared_ptr<MxStream::MxStreamManager> mxStreamManager, int ret) {
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "SendDataAndGetResult failed";
        mxStreamManager->DestroyAllStreams();
    }
    return APP_ERR_OK;
}

int main(int argc, char *argv[]) {
    uint32_t operationMode = SENDDATA_GETRESULT;
    if (argc > 1)
        operationMode = atoi(argv[1]);
    std::string pipelineConfigPath = "./test.pipeline";
    std::string streamName = "test";
    auto mxStreamManager = std::make_shared<MxStream::MxStreamManager>();
    APP_ERROR ret = mxStreamManager->InitManager();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Failed to init Stream manager.";
        return ret;
    }
    ret = mxStreamManager->CreateMultipleStreamsFromFile(pipelineConfigPath);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Failed to init Stream manager.";
        return ret;
    }

    // pipeline 输入输出操作
    if (operationMode == SENDDATA_GETRESULT) {
        // 传入具体的输入插件id
        ret = SendDataAndGetResult(mxStreamManager, streamName, 0, 0);
        DetSendData(mxStreamManager, ret);

        // 传入具体的输入插件名称 appsrc0
        ret = SendDataAndGetResult(mxStreamManager, streamName, "appsrc0", 0);
        DetSendData(mxStreamManager, ret);

        // sendData 的第三种方式
        ret = SendDataAndGetResult(mxStreamManager, streamName, "appsrc0");
        DetSendData(mxStreamManager, ret);

    }

    if (operationMode == SENDDATA_GETRESULT_WITH_UNIQUE_ID) {
        // 传入具体的输入插件id
        ret = SendDataWithUniqueIdAndGetResultWithUniqueId(mxStreamManager, streamName, 0);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "SendDataWithUniqueIdAndGetResultWithUniqueId failed";
            mxStreamManager->DestroyAllStreams();
            return ret;
        }
        // 传入具体的输入插件名称 appsrc0
        ret = SendDataWithUniqueIdAndGetResultWithUniqueId(mxStreamManager, streamName, "appsrc0");
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "SendDataWithUniqueIdAndGetResultWithUniqueId failed";
            mxStreamManager->DestroyAllStreams();
            return ret;
        }
    }
    if (operationMode == SENDPROTOBUFFER_GETPROBUFFER) {
        ret = sendProtobufferAndGetProtobuffer(mxStreamManager, streamName, 0, 0);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "sendProtobufferAndGetProtobuffer failed";
            mxStreamManager->DestroyAllStreams();
            return ret;
        }
    }
}
