#ifndef PICODET_POST_PROCESS_H
#define PICODET_POST_PROCESS_H
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

namespace {
    const float DEFAULT_SCORE_THRESH = 0.4;
    const float DEFAULT_NMS_THRESH = 0.5;
    const uint32_t DEFAULT_STRIDES_NUM = 4;
    const uint32_t DEFAULT_CLASS_NUM = 80;
    const uint32_t REG_MAX = 7;
    const uint32_t TENSOR_SIZE = 8;
    const uint32_t TENSOR_SHAPE_SIZE = 3;
    const uint32_t BOXES_TENSOR = 32;
    const uint32_t BBOX_SIZE = 4;
}

namespace MxBase {
    class PicodetPostProcess: public ObjectPostProcessBase
    {
    public:
        PicodetPostProcess() = default;

        ~PicodetPostProcess() = default;

        PicodetPostProcess(const PicodetPostProcess &other) = default;

        PicodetPostProcess &operator=(const PicodetPostProcess &other);

        APP_ERROR Init(const std::map <std::string, std::shared_ptr<void>> &postConfig) override;

        APP_ERROR DeInit() override;

        APP_ERROR Process(const std::vector <MxBase::TensorBase> &tensors,
                          std::vector <std::vector<MxBase::ObjectInfo>> &objectInfos,
                          const std::vector <MxBase::ResizedImageInfo> &resizedImageInfos = {},
                          const std::map <std::string, std::shared_ptr<void>> &paramMap = {}) override;

    protected:
        bool IsValidTensors(const std::vector <MxBase::TensorBase> &tensors) const;

        APP_ERROR ObjectDetectionOutput(const std::vector <MxBase::TensorBase> &tensors,
                                   std::vector <std::vector<MxBase::ObjectInfo>> &objectInfos,
                                   const std::vector <MxBase::ResizedImageInfo> &resizedImageInfos = {});

        void GetScoreAndLabel(const float *outBuffer, const uint32_t idx,float &score, int &curLabel);

        void GenerateBbox(const float *&bboxInfo, std::pair<int, int> center, int stride,
                          const ResizedImageInfo &resizedImageInfo,
                          ObjectInfo &objectInfo);

        APP_ERROR GetStrides(std::string &strStrides);



    protected:
        float scoreThresh_ = DEFAULT_SCORE_THRESH;
        float nmsThresh_ = DEFAULT_NMS_THRESH;
        uint32_t classNum_ = DEFAULT_CLASS_NUM;
        uint32_t stridesNum_ = DEFAULT_STRIDES_NUM;
        std::vector<float> strides_ = {};
    };
    extern "C" {
    std::shared_ptr<MxBase::PicodetPostProcess> GetObjectInstance();
    }
}
#endif