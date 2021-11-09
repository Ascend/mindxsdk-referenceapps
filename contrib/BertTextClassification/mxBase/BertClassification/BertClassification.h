#ifndef MXBASE_TEXT_CLASSIFICATION_BERTCLASSIFICATION_H
#define MXBASE_TEXT_CLASSIFICATION_BERTCLASSIFICATION_H

#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    std::string modelPath;
    std::string vocabTextPath;
    uint32_t maxLength;
    uint32_t labelNumber;

};

class BertClassification {
public:
    APP_ERROR InitTokenMap(const std::string &vocabTextPath, std::map<std::string, int> &tokenMap);
    APP_ERROR LoadLabels(const std::string &labelPath, std::map<int, std::string> &labelMap);
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &outputs, std::string &label);
    APP_ERROR Process(const std::string &textPath, std::string &label);
    APP_ERROR TextToTensor(const std::string &text, std::vector<MxBase::TensorBase> &inputs);
    APP_ERROR WriteResult(std::string &label);
private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_ = {};
    std::map<int, std::string> labelMap_ = {};
    std::map<std::string, int> tokenMap_ = {};
    uint32_t deviceId_ = 0;
    // Maximum length of input sentence.
    uint32_t maxLength_ = 300;
    // Number of tags for inference results.
    uint32_t labelNumber_ = 5;
};


#endif // MXBASE_TEXT_CLASSIFICATION_BERTCLASSIFICATION_H
