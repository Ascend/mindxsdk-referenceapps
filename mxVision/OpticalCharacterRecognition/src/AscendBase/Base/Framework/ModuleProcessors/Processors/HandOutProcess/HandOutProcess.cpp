#include "HandoutProcess.h"
#include "DbnetPreProcess/DbnetPreProcess.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <regex>

using namespace ascendOCR;

HandoutProcess::HandoutProcess()
{
    withoutInputQueue_ = true;
    isStop_ = false;
}
HandoutProcess::~HandoutProcess() {}

APP_ERROR HandoutProcess::Init(ConfigParser &configParser, ModuleInitParams &initParams)
{
    LogInfo << "Begin to init instance " << initParams.instanceId;
    InitParams(initParams);
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogError << "HandOutProcess[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret);
        return ret;
    }
    LogInfo << "HandOutProcess[" << instanceId_ << "]: Init success.";
    return APP_ERR_OK;
}

APP_ERROR HandoutProcess::DeInit(void)
{
    LogInfo << "HandOutProcess[" << instanceId_ << "]: Deinit success.";
    return APP_ERR_OK;
}

APP_ERROR HandoutProcess::ParseConfig(ConfigParser &configParser)
{
    configParser.GetBoolValue("saveInferResult", saveInferResult);
    if (saveInferResult) {
        configParser.GetStringValue("resultPath", resultPath);
    }
    return APP_ERR_OK;
}

APP_ERROR HandoutProcess::Process(std::shared_prt<void> commonData)
{
    std::string imgConfig = "./data/config/" + pipelineName_;
    LogInfo << pipelineName_;
    std::ifstream imgFileCount;
    imgFileCount.open(imgConfig);
    std::string imgPathCount;
    int imgPath = 0;
    while (getline(imgFile, imgPath) && !Signal::signalRecieved) {
        LogInfo << pipelineName_ << " read file: " << imgPath;
        basename = Utils::BaseName(imgPath);
        std::regex_match(basename.c_str(), m, reg);
        if (m.empty()) {
            LogError << "Please check the image name format of " << basename <<
                ". The image name should be xxx_xxx.xxx";
            continue;
        }
        imgId_++;
        std::shared_ptr<CommonData> data = std::make_shared<CommonData>();
        data->imgPath = imgPath;
        data->imgId = imgId_;
        data->imgTotal = imgTotal;
        data->imgName = basename;
        data->saveFileName = Utils::GenerateResName(imgPath);
        SendToNextModule(MT_DbnetPreProcess, data, data->channelId);
    }
    imgFile.close();
    return APP_ERR_OK;
}