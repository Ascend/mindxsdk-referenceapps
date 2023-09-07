#ifndef CPP_CRNNINFERPROCESS_H
#define CPP_CRNNINFERPROCESS_H

#include "CharacterRecognitionPost.h"

#include "ModuleManagers/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "CommonData/CommonData.h"
#include "Signal.h"
#include "MxBase/MxBase.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"

class CrnnInferProcess : public ascendOCR::ModuleBase {
public:
    CrnnInferProcess();
    ~CrnnInferProcess();
    APP_ERROR Init(ConfigParser &configParser, ascendOCR::ModuleInitParams &initParams);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    int mStdHeight = 48;
    int32_t deviceId_ = 0;
    bool staticMethod = true;
    std::vector<MxBase::Model *> crnnNet_;
    std::vector<uint32_t> batchSizeList;
    std::vector<MxBase::Tensor> crnnOutputs;

    APP_ERROR ParseConfig(ConfigParser &configParser);
    std::vector<MxBase::Tensor> CrnnModelInfer(uint8_t *srcData, uint32_t BatchSize, int maxResizedW);
};

MODULE_REGIST(CrnnInferProcess)

#endif

