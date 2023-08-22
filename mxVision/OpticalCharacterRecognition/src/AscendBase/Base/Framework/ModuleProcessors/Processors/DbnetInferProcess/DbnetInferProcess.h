#ifndef CPP_DBNETINFERPROCESS_H
#define CPP_DBNETINFERPROCESS_H

#include "ModuleManagers/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "CommonData/CommonData.h"
#include "Signal.h"
#include "MxBase/MxBase.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"

class DbnetInferProcess : public ascendOCR::ModuleBase {
public:
    DbnetInferProcess();
    ~DbnetInferProcess();
    APP_ERROR Init(ConfigParser &configParser, ascendOCR::ModuleInitParams &initParams);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    int32_t deviceId_ = 0;
    std::unique_ptr<MxBase::Model> dbNet_;
    std::vector<MxBase::Tensor> dbNetoutputs;

    APP_ERROR ParseConfig(ConfigParser &configParser);
};

MODULE_REGIST(DbnetInferProcess)

#endif

