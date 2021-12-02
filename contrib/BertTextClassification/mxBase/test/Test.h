//
// Created by 13352 on 2021/10/19.
//

#ifndef MXBASE_TEXT_CLASSIFICATION_TEST_H
#define MXBASE_TEXT_CLASSIFICATION_TEST_H


#include <BertClassification.h>

class Test {
public:
    static void InitBertParam(InitParam &initParam);
    static APP_ERROR test_accuracy();
    static APP_ERROR test_input();
};


#endif // MXBASE_TEXT_CLASSIFICATION_TEST_H
