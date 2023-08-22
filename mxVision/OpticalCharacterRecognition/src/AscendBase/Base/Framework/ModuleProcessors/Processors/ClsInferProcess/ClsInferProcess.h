#ifndef CPP_CLSINFERPROCESS_H
#define CPP_CLSINFERPROCESS_H

#include "ModuleManagers/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "Utils.h"
#include "CommonData/CommonData.h"
#include "Signal.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"

class ClsInferProcess : public ascendOCR::ModuleBase {
public:
    ClsInferProcess();
    ~ClsInferProcess();
    APP_ERROR Init(ConfigParser &configParser, ascendOCR::ModuleInitParams &initParams);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    int clsWidth;
    int clsHeight;
    int32_t deviceId_ = 0;
    std::unique_ptr<MxBase::Model> ClsNet_;
    std::vector<uint32_t> batchSizeList;
    std::vector<MxBase::Tensor> ClsOutputs;

    APP_ERROR ParseConfig(ConfigParser &configParser);
    std::vector<MxBase::Tensor> ClsModelInfer(uint8_t *srcData, uint32_t BatchSize, int maxResizedW);
};

MODULE_REGIST(ClsInferProcess)

#endif

