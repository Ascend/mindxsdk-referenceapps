#ifndef CPP_CLSPREPROCESS_H
#define CPP_CLSPREPROCESS_H

#include "ModuleManagers/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "Utils.h"
#include "CommonData/CommonData.h"
#include "Signal.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"

class ClsPreProcess : public ascendOCR::ModuleBase {
public:
    ClsPreProcess();
    ~ClsPreProcess();
    APP_ERROR Init(ConfigParser &configParser, ascendOCR::ModuleInitParams &initParams);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    int clsWidth = 192;
    int clsHeight = 48;
    int32_t deviceId_ = 0;
    std::vector<uint64_t> batchSizeList;

    APP_ERROR ParseConfig(ConfigParser &configParser);
    std::vector<uint32_t> GetClsBatchSize(uint32_t frameSize);
    uint8_t *PreprocessCls(std::vector<cv::Mat> &frames, uint32_t batchSize);
};

MODULE_REGIST(ClsPreProcess)

#endif

