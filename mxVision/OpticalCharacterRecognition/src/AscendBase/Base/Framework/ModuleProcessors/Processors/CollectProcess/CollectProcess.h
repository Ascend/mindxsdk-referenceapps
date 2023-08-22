#ifndef CPP_COLLECTPROCESS_H
#define CPP_COLLECTPROCESS_H

#include <unordered_map>

#include "ModuleManagers/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "CommonData/CommonData.h"
#include "Signal.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"

class CollectProcess : public ascendOCR::ModuleBase {
public:
    CollectProcess();
    ~CollectProcess();
    APP_ERROR Init(ConfigParser &configParser, ascendOCR::ModuleInitParams &initParams);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    std::string resultPath;
    bool saveInferResult = false;

    std::unordered_map<int, int> idMap;
    int inferSize = 0;

    APP_ERROR ParseConfig(ConfigParser &configParser);
    
    void SignalSend(int imgTotal);
};

MODULE_REGIST(CollectProcess)

#endif

