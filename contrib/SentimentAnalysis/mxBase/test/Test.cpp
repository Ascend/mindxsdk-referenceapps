#include "Test.h"
#include <algorithm>
#include <iostream>
#include <fstream>

namespace {
  const uint32_t MAX_LENGTH = 500;
  const uint32_t LABEL_NUMBER = 3;
  const std::string LABEL_LIST[LABEL_NUMBER] = {"消极", "积极", "中性"};
}

void Test::InitBertParam(InitParam &initParam) {
  initParam.deviceId = 0;
  initParam.labelPath = "./model/sentiment_analysis_label.names";
  initParam.modelPath = "./model/sentiment_analysis.om";
  initParam.vocabTextPath = "./data/vocab.txt";
  initParam.maxLength = MAX_LENGTH;
  initParam.labelNumber = LABEL_NUMBER;
}

APP_ERROR Test::test_accuracy() {
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

  // Open test file.
  std::ifstream fin("data/test.csv");
  if (fin.fail()) {
    LogError << "Failed to open csvPath file: test.csv.";
    sentiment_analysis->DeInit();
    return APP_ERR_COMM_OPEN_FAIL;
  }
  std::string line, prediction_label;
  std::vector<std::vector<std::string>> prediction_label_lists;

  // Init prediction_label_lists.
  for (int i = 0;i < LABEL_NUMBER;i++) {
    std::vector<std::string> temp;
    prediction_label_lists.push_back(temp);
  }
  int index = 0, all_num = 0;
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
	  // Remove the end \r.
    text = text.replace(text.find("\r"),1,"");
    // Start inference.
    ret = sentiment_analysis->Process(text, prediction_label);

    index = std::find(LABEL_LIST, LABEL_LIST + LABEL_NUMBER, label) - LABEL_LIST;

    // Determine whether the prediction result is correct.
    if (prediction_label == label){
      prediction_label_lists[index].push_back("true");
    }
    else{
      prediction_label_lists[index].push_back("false");
    }

    all_num++;
    if (ret != APP_ERR_OK) {
      LogError << "SentimentAnalysis process failed, ret=" << ret << ".";
      sentiment_analysis->DeInit();
      return ret;
    }
  }
  endTime = clock();
  std::cout << "The average time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC /  all_num
            << "s" << std::endl;
  sentiment_analysis->DeInit();

  // Calculate accuracy.
  int all_count = 0, count = 0;
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
    accuracy = static_cast<double>(count) / label_list.size();
    std::cout << LABEL_LIST[index] << "类的精确度为：" << accuracy << std::endl;
    index++;
  }
  accuracy = static_cast<double>(all_count) / all_num;
  std::cout << "全部类的精确度为：" << accuracy << std::endl;
  return APP_ERR_OK;
}

APP_ERROR Test::test_input() {
  std::vector<std::string> input_text = {};
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
  std::string text;
  std::ifstream infile;
  // Open text file.
  infile.open("data/test.txt", std::ios_base::in);
  // Check text file validity.
  if (infile.fail()) {
    LogError << "Failed to open textPath file: test.txt.";
    sentiment_analysis->DeInit();
    return APP_ERR_COMM_OPEN_FAIL;
  }
  while (std::getline(infile, text)) {
    for (uint32_t i = 0;i < input_text.size();i++) {
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
  }

  // Destroy.
  sentiment_analysis->DeInit();
  return APP_ERR_OK;
}