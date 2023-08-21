#ifndef CPP_DBNETPREPROCESS_H
#define CPP_DBNETPREPROCESS_H

#include "ModuleManagers/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "CommonData/CommonData.h"
#include "Utils.h"
#include "Signal.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"
class DbnetPreProcess : public ascendOCR::ModuleBase {
public:
    DbnetPreProcess();
    ~DbnetPreProcess();
    APP_ERROR Init(ConfigParser &configParser, ascendOCR::ModuleInitParams &initParams);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    std::string deviceType_;
    int32_t deviceId_ = 0;

    std::unique_ptr<MxBase::ImageProcessor> imageProcessor;
    uint64_t MaxH = 0;
    uint64_t MaxW = 0;

    std::pair<uint64_t, uint64_t> maxDotGear;

    std::vector<float> mean_ = { 0.458f, 0.456f, 0.406f };
    std::vector<float> scale_ = { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };

    std::vector<std::pair<uint64_t, uint64_t>> gearInfo;

    APP_ERROR ParseConfig(ConfigParser &configParser);
    cv::Mat DecodeImgDvpp(std::string imgPath);
    void getMatchedGear(const cv::Mat &inImg, std::pair<uint64_t, uint64_t> &gear);
    void resize(const cv::Mat &inImg, cv::Mat &outImg, const std::pair<uint64_t, uint64_t> &getMatchedGear, float &ratio);
    void padding(cv::Mat &inImg, const std::pair<uint64_t, uint64_t> &gear);
    void normalizeByChannel(std::vector<cv::Mat> &bgr_channels);
};

MODULE_REGIST(DbnetPreProcess)

#enfif

