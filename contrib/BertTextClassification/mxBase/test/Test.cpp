//
// Created by 13352 on 2021/10/19.
//

#include "Test.h"
#include <iostream>
#include <fstream>

namespace {
    const uint32_t MAX_LENGTH = 300;
    const uint32_t LABEL_NUMBER = 5;
    const uint32_t SAMPLE_NUMBER = 99;
    const std::string LABEL_LIST[LABEL_NUMBER] = {"体育", "健康", "军事", "教育", "汽车"};
}

void Test::InitBertParam(InitParam &initParam) {
  initParam.deviceId = 0;
  initParam.labelPath = "./model/bert_text_classification_labels.names";
  initParam.modelPath = "./model/bert_text_classification.om";
  initParam.vocabTextPath = "data/vocab.txt";
  initParam.maxLength = MAX_LENGTH;
  initParam.labelNumber = LABEL_NUMBER;
}

APP_ERROR Test::test_accuracy() {
  InitParam initParam;
  InitBertParam(initParam);
  auto bert = std::make_shared<BertClassification>();

  // Initialize the configuration information required for model inference.
  APP_ERROR ret = bert->Init(initParam);
  if (ret != APP_ERR_OK) {
    LogError << "BertClassification init failed, ret=" << ret << ".";
    bert->DeInit();
    return ret;
  }

  // Open test file.
  std::ifstream fin("data/test.csv");
  std::string line, prediction_label;
  std::vector<std::vector<std::string>> prediction_label_lists;

  // Init prediction_label_lists.
  for (int i = 0;i < LABEL_NUMBER;i++) {
    std::vector<std::string> temp;
    prediction_label_lists.push_back(temp);
  }
  int index = 0, count = 0;
  clock_t startTime,endTime;
  startTime = clock();
  while (getline(fin, line)) {
    std::istringstream sin(line);
    std::string label, text;
    std::string field;
    while (getline(sin, field, ',')) {
      label = field;
      break;
    }
    text = line.substr(line.find_first_of(',') + 1);
    // Remove the start and end ".
    if (text.find("\"") == 0) {
      text = text.replace(text.find("\""),1,"");
      text = text.replace(text.find_last_of("\""),1,"");
    }
    // Start inference.
    ret = bert->Process(text, prediction_label);

    if (count != 0 && count % SAMPLE_NUMBER == 0) {
      index++;
    }

    // Determine whether the prediction result is correct.
    if (prediction_label == label){
      prediction_label_lists[index].push_back("true");
    }
    else{
      prediction_label_lists[index].push_back("false");
    }

    count++;
    if (ret != APP_ERR_OK) {
      LogError << "BertClassification process failed, ret=" << ret << ".";
      bert->DeInit();
      return ret;
    }
  }
  endTime = clock();
  std::cout << "The average time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC /  SAMPLE_NUMBER / LABEL_NUMBER
            << "s" << std::endl;
  bert->DeInit();

  // Calculate accuracy.
  int all_count = 0;
  index = 0;
  double accuracy;
  for (auto label_list : prediction_label_lists) {
    count = 0;
    for (auto label : label_list) {
      if (label == "true") {
        count++;
        all_count++;
      }
    }
    accuracy = static_cast<double>(count) / SAMPLE_NUMBER;
    std::cout << LABEL_LIST[index] << "类的精确度为：" << accuracy << std::endl;
    index++;
  }
  accuracy = static_cast<double>(all_count) / SAMPLE_NUMBER / LABEL_NUMBER;
  std::cout << "全部类的精确度为：" << accuracy << std::endl;
  return APP_ERR_OK;
}

APP_ERROR Test::test_input() {
  std::vector<std::string> input_text;
  InitParam initParam;
  InitBertParam(initParam);
  auto bert = std::make_shared<BertClassification>();
  // Initialize the configuration information required for model inference.
  APP_ERROR ret = bert->Init(initParam);
  if (ret != APP_ERR_OK) {
    LogError << "BertClassification init failed, ret=" << ret << ".";
    bert->DeInit();
    return ret;
  }
  std::string text;
  std::ifstream infile;
  // Open text file.
  infile.open("data/test.txt", std::ios_base::in);
  // Check text file validity.
  if (infile.fail()) {
    LogError << "Failed to open textPath file: test.txt.";
    bert->DeInit();
    return APP_ERR_COMM_OPEN_FAIL;
  }
  while (std::getline(infile, text)) {
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