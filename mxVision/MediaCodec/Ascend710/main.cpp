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

#include <thread>
#include "MxBase/Log/Log.h"
#include "MxBase/ConfigUtil/ConfigUtil.h"
#include "MxStream/DataType/DataHelper.h"
#include "MxStream/Stream/SequentialStream.h"

using namespace MxBase;
using namespace MxStream;

namespace {
const std::string CONFIG_PATH = "./config/setup.config";
const int SIGNAL_CHECK_TIMESTEP = 10000;
bool g_signalQuit = false;
std::mutex g_mutex;
std::vector<std::thread> g_threads;
std::map<std::string, std::shared_ptr<SequentialStream>> g_sequentialStreams;

static void SigHandler(int signal)
{
    if (signal == SIGINT) {
        g_signalQuit = true;
    }
}

struct StreamConfig {
    size_t channelCount;
    std::string deviceId;
    std::vector<std::string> rtspUrls;
    std::string inputVideoFormat;
    std::string outputImageFormat;
    std::string inputFormat;
    std::string outputFormat;
    std::string videoEncodeWidth;
    std::string videoEncodeHeight;
    std::string videoEncodeFpsMode;
    std::string iFrameInterval;
    std::string resizeWidth;
    std::string resizeHeight;
    std::string fpsMode;

    const std::map<std::string, std::string> QueueProperties = {{"max-size-buffers", "50"}};

    std::map<std::string, std::string> GetRtspProperties(size_t chlIdx) const
    {
        std::map<std::string, std::string> properties = {
            {"rtspUrl", rtspUrls[chlIdx]},
            {"channelId", std::to_string(chlIdx)},
            {"fpsMode", fpsMode},
        };
        return properties;
    }

    std::map<std::string, std::string> GetVideoDecoderProperties(size_t chlIdx) const
    {
        std::map<std::string, std::string> properties = {
            {"inputVideoFormat", inputVideoFormat},
            {"outputImageFormat", outputImageFormat}
        };
        return properties;
    }

    std::map<std::string, std::string> GetImageResizeProperties() const
    {
        std::map<std::string, std::string> properties = {
            {"resizeWidth", resizeWidth},
            {"resizeHeight", resizeHeight}
        };
        return properties;
    }

    std::map<std::string, std::string> GetVideoEncodeProperties() const
    {
        std::map<std::string, std::string> properties = {
            {"imageHeight", videoEncodeHeight},
            {"imageWidth", videoEncodeWidth},
            {"inputFormat", inputFormat},
            {"outputFormat", outputFormat},
            {"fps", videoEncodeFpsMode},
            {"iFrameInterval", iFrameInterval}
        };
        return properties;
    }
};

APP_ERROR ParseFromConfig(const std::string &path, StreamConfig &config)
{
    MxBase::ConfigUtil util;
    MxBase::ConfigData configData;
    util.LoadConfiguration(path, configData, MxBase::CONFIGFILE);

    configData.GetFileValueWarn("stream.channelCount", config.channelCount);
    configData.GetFileValueWarn("stream.deviceId", config.deviceId);
    configData.GetFileValueWarn("stream.fpsMode", config.fpsMode);
    configData.GetFileValueWarn("stream.resizeWidth", config.resizeWidth);
    configData.GetFileValueWarn("stream.resizeHeight", config.resizeHeight);
    configData.GetFileValueWarn("VideoDecoder.inputVideoFormat", config.inputVideoFormat);
    configData.GetFileValueWarn("VideoDecoder.outputImageFormat", config.outputImageFormat);
    configData.GetFileValueWarn("VideoEncoder.inputFormat", config.inputFormat);
    configData.GetFileValueWarn("VideoEncoder.outputFormat", config.outputFormat);
    configData.GetFileValueWarn("VideoEncoder.iamgeWidth", config.videoEncodeWidth);
    configData.GetFileValueWarn("VideoEncoder.imageHeight", config.videoEncodeHeight);
    configData.GetFileValueWarn("VideoEncoder.fpsMode", config.videoEncodeFpsMode);
    configData.GetFileValueWarn("VideoEncoder.iFrameInterval", config.iFrameInterval);

    for (size_t i = 0; i < config.channelCount; ++i) {
        auto name = "stream.ch" + std::to_string(i);
        std::string value;
        auto ret = configData.GetFileValue(name, value);
        if (ret != APP_ERR_OK) {
            LogError << "Please check rtsp param.";
            return APP_ERR_COMM_INVALID_PARAM;
        }
        config.rtspUrls.push_back(value);
    }

    if (config.channelCount > config.rtspUrls.size()) {
        LogError << "Please check param.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    return APP_ERR_OK;
}

APP_ERROR CreateSingleStream(const StreamConfig &config, size_t chlIdx)
{
    auto streamName = "stream" + std::to_string(chlIdx);
    std::shared_ptr<SequentialStream> sequentialStream(new SequentialStream(streamName));
    if (sequentialStream == nullptr) {
        return APP_ERR_COMM_FAILURE;
    }
    sequentialStream->SetDeviceId(config.deviceId);
    sequentialStream->Add(PluginNode("mxpi_rtspsrc", config.GetRtspProperties(chlIdx),
        "mxpi_rtspsrc" + std::to_string(chlIdx)));
    sequentialStream->Add(PluginNode("queue", config.QueueProperties));
    sequentialStream->Add(PluginNode("mxpi_videodecoder", config.GetVideoDecoderProperties(chlIdx)));
    sequentialStream->Add(PluginNode("queue", config.QueueProperties));
    sequentialStream->Add(PluginNode("mxpi_imageresize", config.GetImageResizeProperties()));
    sequentialStream->Add(PluginNode("queue", config.QueueProperties));
    sequentialStream->Add(PluginNode("mxpi_videoencoder", config.GetVideoEncodeProperties()));
    sequentialStream->Add(PluginNode("queue", config.QueueProperties));
    sequentialStream->Add(PluginNode("fakesink"));

    auto ret = sequentialStream->Build();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Failed to build stream.";
        return ret;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    g_sequentialStreams[streamName] = sequentialStream;
    return APP_ERR_OK;
}

APP_ERROR CreateMultiStreams(const StreamConfig &config)
{
    g_threads.resize(config.channelCount);
    for (auto i = 0; i < config.channelCount; ++i) {
        g_threads[i] = std::thread(CreateSingleStream, config, i);
    }
    LogInfo << "Totally " << config.channelCount << " streams were created.";

    return APP_ERR_OK;
}

APP_ERROR StopMultiStreams()
{
    std::vector<std::thread> threads(g_sequentialStreams.size());
    int i = 0;
    for (auto iter = g_sequentialStreams.begin(); iter != g_sequentialStreams.end(); ++iter) {
        threads[i] = std::thread([](std::shared_ptr<SequentialStream> stream) {
            stream->Stop();
        }, iter->second);
        ++i;
    }
    for (auto &th : threads) {
        th.join();
    }
    g_sequentialStreams.clear();
    LogInfo << "All streams were stoppped successfully.";
    return APP_ERR_OK;
}
}

int main(int argc, char* argv[])
{
    StreamConfig config;
    APP_ERROR ret = ParseFromConfig(CONFIG_PATH, config);
    if (ret != APP_ERR_OK) {
        LogError << "Parse config file failed.";
        return APP_ERR_COMM_FAILURE;
    }

    ret = CreateMultiStreams(config);
    if (ret != APP_ERR_OK) {
        LogError << "Create Streams failed.";
        return APP_ERR_COMM_FAILURE;
    }

    signal(SIGINT, SigHandler);

    while (!g_signalQuit) {
        usleep(SIGNAL_CHECK_TIMESTEP);
    }

    ret = StopMultiStreams();
    if (ret != APP_ERR_OK) {
        LogError << "Stop Streams failed.";
        return APP_ERR_COMM_FAILURE;
    }

    for (auto &th : g_threads) {
        th.join();
    }

    return 0;
}
