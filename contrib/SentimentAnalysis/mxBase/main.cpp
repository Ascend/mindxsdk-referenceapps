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
#include "SentimentAnalysis.h"
#include "test/Test.h"

namespace {
    const uint32_t MAX_LENGTH = 500;
    const uint32_t LABEL_NUMBER = 3;
    const std::string LABEL_LIST[LABEL_NUMBER] = {"消极", "积极", "中性"};
}
void InitBertParam(InitParam &initParam)
{
  initParam.deviceId = 0;
  initParam.labelPath = "./model/sentiment_analysis_label.names";
  initParam.modelPath = "./model/sentiment_analysis.om";
  initParam.vocabTextPath = "data/vocab.txt";
  initParam.maxLength = MAX_LENGTH;
  initParam.labelNumber = LABEL_NUMBER;
}

int main(int argc, char* argv[]) {
  if (argc <= 1) {
    LogWarn << "Please input text path, such as './SentimentAnalysis ./data/sample.txt'.";
    return APP_ERR_OK;
  }

  InitParam initParam;
  InitBertParam(initParam);
  auto sentiment_analysis = std::make_shared<SentimentAnalysis>();
  // Initialize the configuration information required for model inference.
  APP_ERROR ret = sentiment_analysis->Init(initParam);
  if (ret != APP_ERR_OK) {
    LogError << "SentimentAnalysis init failed, ret=" << ret << ".";
    sentiment_analysis->DeInit();
    return ret;
  }

  std::string textPath = argv[1];
  std::string aa = textPath.substr(textPath.find_last_of("."));
  if (textPath.substr(textPath.find_last_of(".")) != ".txt") {
    LogError << "please input the txt file!";
    sentiment_analysis->DeInit();
    return APP_ERR_COMM_FAILURE;
  }
  std::string text;
  std::ifstream infile;
  // Open label file.
  infile.open(textPath, std::ios_base::in);
  // Check label file validity.
  if (infile.fail()) {
    LogError << "Failed to open textPath file: " << textPath << ".";
    sentiment_analysis->DeInit();
    return APP_ERR_COMM_OPEN_FAIL;
  }
  while (std::getline(infile, text)) {
    std::string label;
    // Inference begin.
    ret = sentiment_analysis->Process(text, label);
    std::cout << "origin text:" << text <<std::endl;
    std::cout << "label:" << label <<std::endl;
    if (ret != APP_ERR_OK) {
      LogError << "SentimentAnalysis process failed, ret=" << ret << ".";
      sentiment_analysis->DeInit();
      return ret;
    }
  }
  if (text == "") {
    LogError << "The sample.txt text is null, please input right text!";
    sentiment_analysis->DeInit();
    return APP_ERR_COMM_FAILURE;
  }
  // Destroy.
  sentiment_analysis->DeInit();
  return APP_ERR_OK;
}
