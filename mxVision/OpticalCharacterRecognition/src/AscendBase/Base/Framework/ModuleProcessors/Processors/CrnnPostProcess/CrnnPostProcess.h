#ifndef CPP_CRNNPOSTPROCESS_H
#define CPP_CRNNPOSTPROCESS_H

#include <unordered_set>

#include "ModuleManagers/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "CommonData/CommonData.h"
#include "MxBase/MxBase.h"
#include "ErrorCode/ErrorCode.h"
#include "Signal.h"
#include "Log/Log.h"

#include "CharacterRecognitionPost.h"

class CrnnPostProcess : public ascendOCR::ModuleBase {
public:
    CrnnPostProcess();
    ~CrnnPostProcess();
    APP_ERROR Init(ConfigParser &configParser, ascendOCR::ModuleInitParams &initParams);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    CharacterRecognitionPost characterRecognitionPost_;
    std::string recDictionary;
    std::string resultPath;
    bool saveInferResult = false;
    
    std::unordered_set<int> idSet;

    APP_ERROR ParseConfig(ConfigParser &configParser);

    APP_ERROR PostProcessCrnn(uint32_t framesSize, std::vector<MxBase::Tensor> &inferOutput,
        std::vector<std::string> &textsInfos);
};

MODULE_REGIST(CrnnPostProcess)

#endif

