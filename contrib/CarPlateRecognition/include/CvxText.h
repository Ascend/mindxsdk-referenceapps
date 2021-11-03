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

 // 代码来源：https://blog.csdn.net/qq_31261509/article/details/83503591

#ifndef OPENCV_CVX_TEXT_HPP_
#define OPENCV_CVX_TEXT_HPP_

#include <ft2build.h>
#include FT_FREETYPE_H
#include "opencv2/opencv.hpp"

class CvxText {

public:
    CvxText(const char* freeType);
    virtual ~CvxText();
    void getFont(int* type, cv::Scalar* size = nullptr, bool* underline = nullptr, float* diaphaneity = nullptr);
    void setFont(int* type, cv::Scalar* size = nullptr, bool* underline = nullptr, float* diaphaneity = nullptr);
    void restoreFont();
    int putText(cv::Mat& img, char* text, cv::Point pos);
    int putText(cv::Mat& img, const wchar_t* text, cv::Point pos);
    int putText(cv::Mat& img, const char* text, cv::Point pos, cv::Scalar color);
    int putText(cv::Mat& img, const wchar_t* text, cv::Point pos, cv::Scalar color);
    int ToWchar(char* src, wchar_t* &dest, const char *locale = "zh_CN.utf8");

private:
    CvxText& operator = (const CvxText&);
    void putWChar(cv::Mat& img, wchar_t wc, cv::Point& pos, cv::Scalar color);
    FT_Library   m_library;   // 字库
    FT_Face      m_face;      // 字体
    int         m_fontType;
    cv::Scalar  m_fontSize;
    bool      m_fontUnderline;
    float     m_fontDiaphaneity;
};

#endif