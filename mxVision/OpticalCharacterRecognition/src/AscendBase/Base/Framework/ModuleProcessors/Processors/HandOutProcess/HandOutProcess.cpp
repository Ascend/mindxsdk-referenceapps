#include "HandOutProcess.h"
#include "DbnetPreProcess/DbnetPreProcess.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <regex>

using namespace ascendOCR;

HandOutProcess::HandOutProcess()
{
    withoutInputQueue_ = true;
    isStop_ = false;
}
HandOutProcess::~HandOutProcess() {}

APP_ERROR HandOutProcess::Init(ConfigParser &configParser, ModuleInitParams &initParams)
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

APP_ERROR HandOutProcess::DeInit(void)
{
    LogInfo << "HandOutProcess[" << instanceId_ << "]: Deinit success.";
    return APP_ERR_OK;
}

APP_ERROR HandOutProcess::ParseConfig(ConfigParser &configParser)
{
    configParser.GetBoolValue("saveInferResult", saveInferResult);
    if (saveInferResult) {
        configParser.GetStringValue("resultPath", resultPath);
    }
    return APP_ERR_OK;
}

APP_ERROR HandOutProcess::Process(std::shared_ptr<void> commonData)
{
    std::string imgConfig = "./data/config/" + pipelineName_;
    LogInfo << pipelineName_;
    std::ifstream imgFileCount;
    imgFileCount.open(imgConfig);
    std::string imgPathCount;
    int imgTotal = 0;
    while (getline(imgFileCount, imgPathCount)) {
        imgTotal++;
    }
    imgFileCount.close();
    std::ifstream imgFile;
    imgFile.open(imgConfig);
    std::string imgPath;
    std::regex reg("^([A-Za-z]+)_([0-9+].*$)");
    std::cmatch m;
    std::string basename;
    while (getline(imgFile, imgPath) && !Signal::signalRecieved) {
        LogInfo << pipelineName_ << " read file:" << imgPath;
        basename = Utils::BaseName(imgPath);
        std::regex_match(basename.c_str(), m, reg);
        if (m.empty()) {
            LogError << "Please check the image name format of " << basename <<
                ". the image name should be xxx_xxx.xxx";
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
