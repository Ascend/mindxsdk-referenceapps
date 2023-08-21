#ifndef CPP_DBNETPOSTPROCESS_H
#define CPP_DBNETPOSTPROCESS_H

#include "ModuleManagers/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "CommonData/CommonData.h"
#include "ErrorCode/ErrorCode.h"
#include "Signal.h"
#include "Log/Log.h"

class DbnetPostProcess : public ascendOCR::ModuleBase {
public:
    DbnetPostProcess();
    ~DbnetPostProcess();
    APP_ERROR Init(ConfigParser &configParser, ascendOCR::ModuleInitParams &initParams);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    bool saveInferResult = false;
    std::string resultPath;
    std::string nextModule;

    APP_ERROR ParseConfig(ConfigParser &configParser);
    
    static float CalcCropWidth(const TextObjectInfo &textObject);
    static float CalcCropHeight(const TextObjectInfo &textObject);
};

MODULE_REGIST(DbnetPostProcess)

#enfif

