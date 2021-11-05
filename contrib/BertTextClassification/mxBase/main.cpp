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

#include <iostream>
#include <fstream>
#include "BertClassification.h"
#include "test/Test.h"

namespace {
    const uint32_t MAX_LENGTH = 300;
    const uint32_t LABEL_NUMBER = 5;
    const uint32_t SAMPLE_NUMBER = 99;
    const std::string LABEL_LIST[LABEL_NUMBER] = {"体育", "健康", "军事", "教育", "汽车"};
}

void InitBertParam(InitParam &initParam) {
  initParam.deviceId = 0;
  initParam.labelPath = "./model/bert_text_classification_labels.names";
  initParam.modelPath = "./model/bert_text_classification.om";
  initParam.vocabTextPath = "data/vocab.txt";
  initParam.maxLength = MAX_LENGTH;
  initParam.labelNumber = LABEL_NUMBER;
}

int main(int argc, char* argv[]) {
  if (argc <= 1) {
    LogWarn << "Please input text path, such as './mxBase_text_classification ./data/sample.txt'.";
    return APP_ERR_OK;
  }

  InitParam initParam;
  InitBertParam(initParam);
  auto bert = std::make_shared<BertClassification>();
  // Initialize the configuration information required for model inference.
  APP_ERROR ret = bert->Init(initParam);
  if (ret != APP_ERR_OK) {
    LogError << "BertClassification init failed, ret=" << ret << ".";
    return ret;
  }

  std::string textPath = argv[1];
  std::string aa = textPath.substr(textPath.find_last_of("."));
  if (textPath.substr(textPath.find_last_of(".")) != ".txt") {
    LogError << "please input the txt file!";
    return APP_ERR_COMM_FAILURE;
  }
  std::string text;
  std::ifstream infile;
  // Open text file.
  infile.open(textPath, std::ios_base::in);
  // Check text file validity.
  if (infile.fail()) {
    LogError << "Failed to open textPath file: " << textPath << ".";
    return APP_ERR_COMM_OPEN_FAIL;
  }
  while (std::getline(infile, text)) {
    if (text == "") {
      LogError << "The sample.txt text is null, please input right text!";
      return APP_ERR_COMM_FAILURE;
    }
    std::string label;
    // Inference begin.
    ret = bert->Process(text, label);
    std::cout << "origin text:" << text <<std::endl;
    std::cout << "label:" << label <<std::endl;
    if (ret != APP_ERR_OK) {
      LogError << "BertClassification process failed, ret=" << ret << ".";
      bert->DeInit();
      return ret;
    }
  }
  // Destroy.
  bert->DeInit();
  return APP_ERR_OK;
}
