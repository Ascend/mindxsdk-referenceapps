#ifndef CPP_CRNNPREPROCESS_H
#define CPP_CRNNPREPROCESS_H

#include "ModuleManagers/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "Utils.h"
#include "CommonData/CommonData.h"
#include "Signal.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"

class CrnnPreProcess : public ascendOCR::ModuleBase {
public:
    CrnnPreProcess();
    ~CrnnPreProcess();
    APP_ERROR Init(ConfigParser &configParser, ascendOCR::ModuleInitParams &initParams);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    int mStdHeight = 48;
    int recMinWidth = 320;
    int recMaxWidth = 2240;
    bool staticMethod = true;
    bool isClassification = false;
    std::vector<std::pair<uint64_t, uint64_t>> gearInfo;
    std::vector<uint64_t> batchSizeList;

    APP_ERROR ParseConfig(ConfigParser &configParser);

    std::vector<uint32_t> GetCrnnBatchSize(uint32_t frameSize);

    int GetCrnnMaxWidth(std::vector<cv::Mat> frames, float maxWHRatio);

    uint8_t *PreprocessCrnn(std::vector<cv::Mat> &frames, uint32_t batchSize, int maxResizeW, float maxWHRatio,
        std::vector<ResizedImageInfo> &resizedImageInfos);

    void GetGearInfo(int maxResizedW, std::pair<uint64_t, uint64_t> &gear);
};

MODULE_REGIST(CrnnPreProcess)

#endif

