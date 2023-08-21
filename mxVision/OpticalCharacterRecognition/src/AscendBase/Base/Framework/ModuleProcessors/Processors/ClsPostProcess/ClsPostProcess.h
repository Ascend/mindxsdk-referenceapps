#ifndef CPP_CLSPOSTPROCESS_H
#define CPP_CLSPOSTPROCESS_H

#inclued <unordered_set>

#include "ModuleManagers/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "CommonData/CommonData.h"
#include "MxBase/MxBase.h"
#include "ErrorCode/ErrorCode.h"
#include "Signal.h"
#include "Log/Log.h"

class ClsPostProcess : public ascendOCR::ModuleBase {
public:
    ClsPostProcess();
    ~ClsPostProcess();
    APP_ERROR Init(ConfigParser &configParser, ascendOCR::ModuleInitParams &initParams);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR PostProcessCls(uint32_t frameSize, std::vector<MxBase::Tensor> &inferOutput,
        std::vector<cv::Mat> &imgMatVec);
};

MODULE_REGIST(ClsPostProcess)

#enfif

