#ifndef CPP_HANDOUTPROCESS_H
#define CPP_HANDOUTPROCESS_H

#include "ModuleManagers/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "CommonData/CommonData.h"
#include "Utils.h"
#include "Signal.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"

class HandoutProcess : public ascendOCR::ModuleBase {
public:
    HandoutProcess();
    ~HandoutProcess();
    APP_ERROR Init(ConfigParser &configParser, ascendOCR::ModuleInitParams &initParams);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    int imgId_ = 0;
    std::string deviceType_;
    APP_ERROR ParseConfig(ConfigParser &configParser);
    bool saveInferResult;
    std::string resultPath;
};
MODULE_REGIST(HandOutProcess)
#endif

