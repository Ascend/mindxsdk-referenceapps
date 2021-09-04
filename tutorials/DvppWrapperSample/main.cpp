/*
* Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
* Description: Interface sample of DvppWrapper
* Author: MindX SDK
* Create: 2021
* History: NA
*/

#include <iostream>
#include <string>
#include <fstream>
#include <condition_variable>
#include <map>
#include "opencv2/opencv.hpp"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

using namespace std;

namespace {
    typedef char* char_str;
    using namespace MxBase;

    const uint32_t ENCODE_TEST_DEVICE_ID = 1;
    const uint32_t ENCODE_IMAGE_HEIGHT = 1080;
    const uint32_t ENCODE_IMAGE_WIDTH = 1920;
    const uint32_t ENCODE_FRAME_INTERVAL = 25;
    const uint32_t MAX_FRAME_COUNT = 300;
    uint32_t g_callTime = MAX_FRAME_COUNT;
    FILE *g_fp = nullptr;

    std::shared_ptr<DvppWrapper> g_dvppCommon;
    std::shared_ptr<DvppWrapper> dvppImageDecodeWrapper;
    DeviceContext deviceContext_ = {};

    std::string ReadFileContent(const std::string& filePath);
    APP_ERROR DeInitResource();
    APP_ERROR DeInitDevice();
    APP_ERROR InitDevice();
    APP_ERROR InitResource();
    APP_ERROR TestVpcResizeNormal();
    APP_ERROR TestDvppJpegDecodeNormal();
    APP_ERROR TestVpcCropNormal();
    APP_ERROR DvppEncodeInit();
    APP_ERROR DvppEncodeProcess(std::string file);
    APP_ERROR DvppEncodeDeInit();
    APP_ERROR TestDvppVencNormal();
    APP_ERROR RunTest();

    APP_ERROR InitDevice()
    {
        APP_ERROR result = APP_ERR_OK;
        result = DeviceManager::GetInstance()->InitDevices();
        if (result != APP_ERR_OK) {
            return result;
        }
        deviceContext_.devId = ENCODE_TEST_DEVICE_ID;
        result = DeviceManager::GetInstance()->SetDevice(deviceContext_);
        if (result != APP_ERR_OK) {
            return result;
        }
        return result;
    }
    APP_ERROR DeInitDevice()
    {
        APP_ERROR result = DeviceManager::GetInstance()->DestroyDevices();
        if (result != APP_ERR_OK) {
        }
        return result;
    }

    APP_ERROR InitResource()
    {
        APP_ERROR ret = APP_ERR_OK;
        g_dvppCommon = std::make_shared<DvppWrapper>();
        if (g_dvppCommon == nullptr) {
            LogError << "Failed to create g_dvppCommon object";
            return APP_ERR_COMM_INIT_FAIL;
        }
        LogInfo << "DvppCommon object created successfully";
        ret = g_dvppCommon->Init();
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Failed to initialize g_dvppCommon object.";
            return ret;
        }
        LogInfo << "DvppCommon object initialized successfully";
        return APP_ERR_OK;
    }

    APP_ERROR DeInitResource()
    {
        APP_ERROR ret = APP_ERR_OK;
        ret = g_dvppCommon->DeInit();
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Failed to deInit g_dvppCommon object.";
            return ret;
        }
        LogInfo << "DvppCommon object deInit successfully";
        return ret;
    }

    std::string ReadFileContent(const std::string& filePath)
    {
        std::ifstream file(filePath, std::ios::binary);
        if (!file) {
            LogError << "Invalid file. filePath(" << filePath << ")";
            return "";
        }
        file.seekg(0, std::ifstream::end);
        uint32_t fileSize = file.tellg();
        file.seekg(0);
        std::vector<char> buffer;
        buffer.resize(fileSize);
        file.read(buffer.data(), fileSize);
        file.close();
        return std::string(buffer.data(), fileSize);
    }

    APP_ERROR TestVpcResizeNormal()
    {
        int resizeWidth = 240;
        int resizeHeight = 100;
        std::string filepath = "./test5.jpg";
        DvppDataInfo input, output;
        APP_ERROR ret = g_dvppCommon->DvppJpegDecode(filepath, input);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to decode file : " << filepath;
            return ret;
        }
        ResizeConfig config;
        config.width = resizeWidth;
        config.height = resizeHeight;
        ret = g_dvppCommon->VpcResize(input, output, config);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to resize file : " << filepath;
            return ret;
        }
        input.destory(input.data);
        // save pic
        DvppDataInfo dataInfo;
        const uint32_t level = 100;
        ret = g_dvppCommon->DvppJpegEncode(output, dataInfo, level);
        if (ret != APP_ERR_OK) {
            return ret;
        }
        MemoryData data(dataInfo.dataSize, MemoryData::MEMORY_HOST);
        MemoryData src(static_cast<void*>(dataInfo.data), dataInfo.dataSize, MemoryData::MEMORY_DVPP);
        ret = MemoryHelper::MxbsMallocAndCopy(data, src);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to copy data to host";
            return ret;
        }
        FILE *fp = fopen("./write_result.jpg", "w");
        if (fp == nullptr) {
            LogError << "open file fail";
        }
        fwrite(data.ptrData, 1, data.size, fp);
        fclose(fp);
        output.destory(output.data);
        dataInfo.destory(dataInfo.data);
        data.free(data.ptrData);
        return ret;
    }

    APP_ERROR TestDvppJpegDecodeNormal()
    {
        std::string filepath = "./test5.jpg";
        DvppDataInfo output;
        APP_ERROR ret = g_dvppCommon->DvppJpegDecode(filepath, output);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to decode file: " << filepath;
            return ret;
        }
        MemoryData des(output.dataSize, MemoryData::MEMORY_HOST);
        MemoryData src(static_cast<void*>(output.data), output.dataSize, MemoryData::MEMORY_DVPP);
        ret = MemoryHelper::MxbsMallocAndCopy(des, src);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to copy data to host";
            return ret;
        }
        std::string result(static_cast<char_str>(des.ptrData), des.size);
        std::string content = ReadFileContent("./decode.jpg");
        if (result == content) {
            LogInfo << "Decode success";
        } else {
            LogInfo << "Decode incorrect";
        }
        output.destory(output.data);
        des.free(des.ptrData);
        return ret;
    }

    APP_ERROR TestVpcCropNormal()
    {
        std::string filepath = "./test5.jpg";
        DvppDataInfo input;
        APP_ERROR ret = g_dvppCommon->DvppJpegDecode(filepath, input);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to decode file: " << filepath;
            return ret;
        }
        DvppDataInfo output;
        CropRoiConfig config{22, 226, 230, 30};
        ret = g_dvppCommon->VpcCrop(input, output, config);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to crop file: " << filepath;
            return ret;
        }
        input.destory(input.data);
        // save pic
        DvppDataInfo encodeInfo;
        const uint32_t level = 100;
        ret = g_dvppCommon->DvppJpegEncode(output, encodeInfo, level);
        if (ret != APP_ERR_OK) {
            return ret;
        }
        MemoryData data(encodeInfo.dataSize, MemoryData::MEMORY_HOST);
        MemoryData src(static_cast<void*>(encodeInfo.data), encodeInfo.dataSize, MemoryData::MEMORY_DVPP);
        ret = MemoryHelper::MxbsMallocAndCopy(data, src);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to copy data to host";
            return ret;
        }
        FILE *fp = fopen("./write_result_crop.jpg", "w");
        fwrite(data.ptrData, 1, data.size, fp);
        fclose(fp);
        output.destory(output.data);
        encodeInfo.destory(encodeInfo.data);
        data.free(data.ptrData);
        return ret;
    }

    APP_ERROR DvppEncodeInit()
    {
        VencConfig vencConfig = {};
        vencConfig.deviceId = ENCODE_TEST_DEVICE_ID;
        vencConfig.height = ENCODE_IMAGE_HEIGHT;
        vencConfig.width = ENCODE_IMAGE_WIDTH;
        vencConfig.keyFrameInterval = ENCODE_FRAME_INTERVAL;
        vencConfig.outputVideoFormat = MxBase::MXBASE_STREAM_FORMAT_H264_MAIN_LEVEL;
        vencConfig.inputImageFormat = MxBase::MXBASE_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
        vencConfig.stopEncoderThread = false;
        APP_ERROR ret = g_dvppCommon->InitVenc(vencConfig);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to initialize g_dvppCommon object.";
            return ret;
        }

        dvppImageDecodeWrapper = make_shared<DvppWrapper>();
        if (dvppImageDecodeWrapper == nullptr) {
            LogError << "Failed to create dvppImageDecodeWrapper object";
            return APP_ERR_COMM_INIT_FAIL;
        }
        ret = dvppImageDecodeWrapper->Init();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to create dvppImageDecodeWrapper object";
            return ret;
        }
        LogInfo << "DvppCommon object initialized successfully";
        return APP_ERR_OK;
    }

    APP_ERROR DvppEncodeProcess(std::string file)
    {
        std::mutex mutex = {};
        std::condition_variable endCond = {};
        std::unique_lock<std::mutex> lock(mutex);
        g_fp = fopen("./test.h264", "wb");
        if (g_fp == nullptr) {
            LogError << "fopen fail";
            return APP_ERR_COMM_INIT_FAIL;
        }

        DvppDataInfo imageDataInfo = {};
        APP_ERROR ret = g_dvppCommon->DvppJpegDecode(file, imageDataInfo);
        if (ret != APP_ERR_OK) {
            LogError << "DvppJpegDecode error";
            return ret;
        }
        using HandleFunction = std::function<void(std::shared_ptr<unsigned char>, unsigned int)>;
        HandleFunction func = [&endCond] (std::shared_ptr<uint8_t> data, uint32_t streamSize) {
            if (data.get() == nullptr) {
                LogError << "data is invaild";
            } else if (streamSize == 0) {
                LogError << "data size is equal to 0";
            } else {
                MemoryData des(streamSize, MemoryData::MEMORY_HOST);
                MemoryData src(static_cast<void*>(data.get()), streamSize, MemoryData::MEMORY_DVPP);
                APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(des, src);
                if (ret != APP_ERR_OK) {
                    LogError << "MxbsMallocAndCopy error";
                }
                fwrite(des.ptrData, 1, des.size, g_fp);

                des.free(des.ptrData);
            }
            g_callTime = g_callTime - 1;
            LogInfo << "call time : " << g_callTime;
        };
        for (uint32_t i = 0; i < MAX_FRAME_COUNT; i++) {
            ret = g_dvppCommon->DvppVenc(imageDataInfo, &func);
            if (ret != APP_ERR_OK) {
                LogError << "DvppVenc error";
                return ret;
            }
        }

        imageDataInfo.destory(imageDataInfo.data);
        while (g_callTime <= 0) {;}

        return APP_ERR_OK;
    }

    APP_ERROR DvppEncodeDeInit()
    {
        if (g_dvppCommon.get() == nullptr) {
            return APP_ERR_COMM_FAILURE;
        }
        APP_ERROR ret = g_dvppCommon->DeInitVenc();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to initialize dvpp encode wrapper Deinit error";
            return ret;
        }
        ret = dvppImageDecodeWrapper->DeInit();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to initialize dvppImageDecodeWrapper Deinit error";
            return ret;
        }
        return APP_ERR_OK;
    }

    APP_ERROR TestDvppVencNormal()
    {
        APP_ERROR ret = DvppEncodeInit();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to DvppEncodeInit";
            return ret;
        }
        ret = DvppEncodeProcess("./test_venc.jpg");
        if (ret != APP_ERR_OK) {
            LogError << "Failed to DvppEncodeProcess";
            return ret;
        }
        ret = DvppEncodeDeInit();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to DvppEncodeDeInit";
            return ret;
        }
        LogInfo << "DvppVencNormal successfully";
        return ret;
    }

    APP_ERROR RunTest()
    {
        APP_ERROR ret = InitDevice();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to InitDevice";
            return ret;
        }
        ret = InitResource();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to InitResource";
            return ret;
        }

        ret = TestVpcResizeNormal();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to TestVpcResizeNormal";
            return ret;
        }
        ret = TestDvppJpegDecodeNormal();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to TestDvppJpegDecodeNormal";
            return ret;
        }
        ret = TestVpcCropNormal();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to TestVpcCropNormal";
            return ret;
        }
        ret = TestDvppVencNormal();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to TestDvppVencNormal";
            return ret;
        }

        ret = DeInitResource();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to DeInitResource";
            return ret;
        }
        ret = DeInitDevice();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to DeInitDevice";
            return ret;
        }

        fclose(g_fp);
        g_fp = nullptr;

        LogInfo << "Run DvppWrapperSample successfully";
        return ret;
    }
}

int main()
{
    RunTest();
    return 0;
}
