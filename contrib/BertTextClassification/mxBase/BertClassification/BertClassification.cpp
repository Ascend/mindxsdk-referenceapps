/*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "BertClassification.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include <unistd.h>
#include <fstream>

namespace {
  const float TEXT_START_CODE = 101.0;
  const float TEXT_END_CODE = 102.0;
  const float TEXT_NOT_FOUND_CODE = 100.0;
  const int FLOAT32_BYTES = 4;
}

// Initialize the dictionary of character encoding.
APP_ERROR BertClassification::InitTokenMap(const std::string &vocabTextPath, std::map<std::string, int> &tokenMap) {
  const std::string text;
  std::ifstream infile;
  // Open label file.
  infile.open(vocabTextPath, std::ios_base::in);
  if(!infile.is_open ()) {
    std::cout << "Open " << vocabTextPath << " file failure!" << std::endl;
    return APP_ERR_COMM_OPEN_FAIL;
  }

  std::string s;
  int count = 0;
  while (std::getline(infile, s)) {
    tokenMap.insert(std::pair<std::string, int>(s, count));
    count++;
  }
  infile.close();
  return APP_ERR_OK;
}

// Load the label file.
APP_ERROR BertClassification::LoadLabels(const std::string &labelPath, std::map<int, std::string> &labelMap) {
  std::ifstream infile;
  // Open label file.
  infile.open(labelPath, std::ios_base::in);
  std::string s;
  // Check label file validity.
  if (infile.fail()) {
    LogError << "Failed to open label file: " << labelPath << ".";
    return APP_ERR_COMM_OPEN_FAIL;
  }
  labelMap.clear();
  // Construct label map.
  int count = 0;
  while (std::getline(infile, s)) {
    size_t eraseIndex = s.find_last_not_of("\r\n\t");
    if (eraseIndex != std::string::npos) {
      s.erase(eraseIndex + 1, s.size() - eraseIndex);
    }
    labelMap.insert(std::pair<int, std::string>(count, s));
    count++;
  }
  infile.close();
  return APP_ERR_OK;
}

APP_ERROR BertClassification::Init(const InitParam &initParam) {
  deviceId_ = initParam.deviceId;
  maxLength_ = initParam.maxLength;
  labelNumber_ = initParam.labelNumber;
  APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
  if (ret != APP_ERR_OK) {
    LogError << "Init devices failed, ret=" << ret << ".";
    return ret;
  }
  ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
  if (ret != APP_ERR_OK) {
    LogError << "Set context failed, ret=" << ret << ".";
    return ret;
  }
  // Load TokenMap.
  ret = InitTokenMap(initParam.vocabTextPath, tokenMap_);
  if (ret != APP_ERR_OK) {
    LogError << "Failed to load tokenMap, ret=" << ret << ".";
    return ret;
  }
  model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
  ret = model_->Init(initParam.modelPath, modelDesc_);
  if (ret != APP_ERR_OK) {
    LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
    return ret;
  }

  // Load labels from file.
  ret = LoadLabels(initParam.labelPath, labelMap_);
  if (ret != APP_ERR_OK) {
    LogError << "Failed to load labels, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR BertClassification::DeInit() {
  model_->DeInit();
  MxBase::DeviceManager::GetInstance()->DestroyDevices();
  return APP_ERR_OK;
}

std::vector<std::string> split_chinese(std::string s) {
  std::vector<std::string> words;
  for (size_t i = 0; i < s.length();) {
    int cplen = 1;
    // The following if-statements are referred to https://en.wikipedia.org/wiki/UTF-8#Description.
    if ((s[i] & 0xf8) == 0xf0)
      cplen = 4;
    else if ((s[i] & 0xf0) == 0xe0)
      cplen = 3;
    else if ((s[i] & 0xe0) == 0xc0)
      cplen = 2;
    if ((i + cplen) > s.length())
      cplen = 1;
    words.push_back(s.substr(i, cplen));
    i += cplen;
  }
  return words;
}

APP_ERROR BertClassification::TextToTensor(const std::string &text, std::vector<MxBase::TensorBase> &inputs) {
  MxBase::TensorBase tensor1, tensor2;
  std::vector<uint32_t> shape = {1, maxLength_};
  std::map<std::string, int>::iterator iter;
  uint32_t i, value, size, end_index;
  std::vector<std::string> words = split_chinese(text);

  float* tensor1Data = new float [maxLength_];
  // Init data.
  for (i = 0;i < maxLength_; i++) {
    tensor1Data[i] = 0.0;
  }

  size = words.size();
  if (size > maxLength_) {
    // Remove start and end characters, length is 2.
    size = maxLength_ - 2;
    end_index = maxLength_ - 1;
  } else {
    end_index = size - 1;
  }

  tensor1Data[0] = TEXT_START_CODE;
  // Text decode.
  for (i = 0;i < size; i++) {
    iter = tokenMap_.find(words[i]);
    if (iter != tokenMap_.end()) {
      value = iter->second;
      tensor1Data[i+1] = float(value);
    } else {
      tensor1Data[i+1] = TEXT_NOT_FOUND_CODE;
    }
  }
  tensor1Data[end_index] = TEXT_END_CODE;

  MxBase::MemoryData memoryData1((void*)tensor1Data, maxLength_ * FLOAT32_BYTES,
                                MxBase::MemoryData::MemoryType::MEMORY_HOST_NEW, deviceId_);
  MxBase::MemoryData deviceData1(maxLength_ * FLOAT32_BYTES,MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
  // Move data from Host to Device.
  APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(deviceData1, memoryData1);
  if (ret != APP_ERR_OK) {
    LogError << "Failed to MxbsMallocAndCopy";
    return ret;
  }
  tensor1 = MxBase::TensorBase(deviceData1, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
  inputs.push_back(tensor1);

  // Init second tensor.
  float* tensor2Data = new float [maxLength_];
  for (i = 0;i < maxLength_; i++) {
    tensor2Data[i] = 0.0;
  }
  MxBase::MemoryData memoryData2((void*)tensor2Data, maxLength_ * FLOAT32_BYTES,
                                MxBase::MemoryData::MemoryType::MEMORY_HOST_NEW, deviceId_);
  MxBase::MemoryData deviceData2(maxLength_ * FLOAT32_BYTES,MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
  ret = MxBase::MemoryHelper::MxbsMallocAndCopy(deviceData2, memoryData2);
  if (ret != APP_ERR_OK) {
    LogError << "Failed to MxbsMallocAndCopy";
    return ret;
  }
  tensor2 = MxBase::TensorBase(deviceData2, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
  inputs.push_back(tensor2);
  // Release memory.
  delete[] tensor1Data;
  delete[] tensor2Data;
  return APP_ERR_OK;
}

APP_ERROR BertClassification::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                        std::vector<MxBase::TensorBase> &outputs) {
  auto dtypes = model_->GetOutputDataType();
  std::vector<uint32_t> shape = {1, labelNumber_};

  MxBase::TensorBase tensor(shape, dtypes[0], MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                            deviceId_);
  APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
  if (ret != APP_ERR_OK) {
    LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
    return ret;
  }
  // Put tensor into outputs.
  outputs.push_back(tensor);

  MxBase::DynamicInfo dynamicInfo = {};
  // Set type STATIC_BATCH
  dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
  ret = model_->ModelInference(inputs, outputs, dynamicInfo);

  if (ret != APP_ERR_OK) {
    LogError << "ModelInference failed, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR BertClassification::PostProcess(const std::vector<MxBase::TensorBase> &outputs, std::string &label) {
  uint32_t i;
  uint32_t maxIndex = 0;
  float maxValue = 0;
  MxBase::TensorBase outTensor = outputs[0];
  outTensor.ToHost();
  // Inference result is tensor(1*5).
  float* inferResult = (float *)outTensor.GetBuffer();
  // Find the category with the highest probability.
  for (i = 0;i < labelNumber_;i++) {
    if (inferResult[i] > maxValue) {
      maxValue = inferResult[i];
      maxIndex = i;
    }
  }
  // Get label.
  label = labelMap_.at(maxIndex);
  return APP_ERR_OK;
}

APP_ERROR BertClassification::WriteResult(std::string &label) {
  std::ofstream ofs;
  ofs.open("out/prediction_label.txt", std::ios::out);
  ofs << label;
  ofs.close();
  return APP_ERR_OK;
}

APP_ERROR BertClassification::Process(const std::string &text, std::string &label) {
  std::vector<MxBase::TensorBase> inputs = {};
  std::vector<MxBase::TensorBase> outputs = {};
  // Convert text to tensor.
  APP_ERROR ret = TextToTensor(text, inputs);
  if (ret != APP_ERR_OK) {
    LogError << "ReadText failed, ret=" << ret << ".";
    return ret;
  }

  ret = Inference(inputs, outputs);
  if (ret != APP_ERR_OK) {
    LogError << "Inference failed, ret=" << ret << ".";
    return ret;
  }

  // Get classification results.
  ret = PostProcess(outputs, label);
  if (ret != APP_ERR_OK) {
    LogError << "PostProcess failed, ret=" << ret << ".";
    return ret;
  }

  // Write results to file.
  ret = WriteResult(label);
  if (ret != APP_ERR_OK) {
    LogError << "Save result failed, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}
