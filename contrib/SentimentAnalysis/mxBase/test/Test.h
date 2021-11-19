#ifndef MXBASE_SENTIMENT_ANALYSIS_TEST_H
#define MXBASE_SENTIMENT_ANALYSIS_TEST_H


#include <SentimentAnalysis.h>

class Test {
public:
    static void InitBertParam(InitParam &initParam);
    static APP_ERROR test_accuracy();
    static APP_ERROR test_input();
};


#endif // MXBASE_SENTIMENT_ANALYSIS_TEST_H
