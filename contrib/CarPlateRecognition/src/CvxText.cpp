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

#include <wchar.h>
#include <assert.h>
#include <locale.h>
#include <ctype.h>
#include <cmath>

#include "CvxText.h"


/* @brief: 打开字库
   @param：freeType：字体ttl文件
   @retval:none
*/
CvxText::CvxText(const char* freeType)
{
    assert(freeType != NULL);

    // 打开字库文件, 创建一个字体
    if(FT_Init_FreeType(&m_library)) throw;
    if(FT_New_Face(m_library, freeType, 0, &m_face)) throw;

    // 设置字体输出参数
    restoreFont();

    // 设置C语言的字符集环境
    setlocale(LC_ALL, "");
}


/* @brief: 释放FreeType资源
   @param：none
   @retval:none
*/
CvxText::~CvxText()
{
    FT_Done_Face(m_face);
    FT_Done_FreeType(m_library);
}


/* @brief: 获取字体.目前有些参数尚不支持
   @param：font：字体类型, 目前不支持
   @param：size：字体大小/空白比例/间隔比例/旋转角度
   @param：underline：下画线
   @param：diaphaneity：透明度
   @retval:none
*/
void CvxText::getFont(int* type, cv::Scalar* size, bool* underline, float* diaphaneity)
{
    if (type) *type = m_fontType;
    if (size) *size = m_fontSize;
    if (underline) *underline = m_fontUnderline;
    if (diaphaneity) *diaphaneity = m_fontDiaphaneity;
}


/* @brief: 设置字体.目前有些参数尚不支持.
   @param：font：字体类型, 目前不支持
   @param：size：字体大小/空白比例/间隔比例/旋转角度
   @param：underline：下画线
   @param：diaphaneity：透明度
   @retval:none
*/
void CvxText::setFont(int* type, cv::Scalar* size, bool* underline, float* diaphaneity)
{
    // 参数合法性检查
    if (type) {
        if(type >= 0) m_fontType = *type;
    }
    if (size) {
        m_fontSize.val[0] = std::fabs(size->val[0]);
        m_fontSize.val[1] = std::fabs(size->val[1]);
        m_fontSize.val[2] = std::fabs(size->val[2]);
        m_fontSize.val[3] = std::fabs(size->val[3]);
    }
    if (underline) {
        m_fontUnderline   = *underline;
    }
    if (diaphaneity) {
        m_fontDiaphaneity = *diaphaneity;
    }

    FT_Set_Pixel_Sizes(m_face, (int)m_fontSize.val[0], 0);
}


/* @brief: 恢复原始的字体设置
   @param：none
   @retval:none
*/
void CvxText::restoreFont()
{
    m_fontType = 0;            // 字体类型(不支持)

    m_fontSize.val[0] = 20;      // 字体大小
    m_fontSize.val[1] = 0.5;   // 空白字符大小比例
    m_fontSize.val[2] = 0.1;   // 间隔大小比例
    m_fontSize.val[3] = 0;      // 旋转角度(不支持)

    m_fontUnderline   = false;   // 下画线(不支持)

    m_fontDiaphaneity = 1.0;   // 色彩比例(可产生透明效果)

    // 设置字符大小
    FT_Set_Pixel_Sizes(m_face, (int)m_fontSize.val[0], 0);
}


/* @brief: 输出汉字(颜色默认为黑色).遇到不能输出的字符将停止.
   @param：img：输出的影象
   @param：text：文本内容
   @param：pos：文本位置
   @retval:返回成功输出的字符长度，失败返回-1.
*/
int CvxText::putText(cv::Mat& img, char* text, cv::Point pos)
{
    return putText(img, text, pos, CV_RGB(255, 255, 255));
}


/* @brief: 输出汉字(颜色默认为黑色).遇到不能输出的字符将停止.
   @param：img：输出的影象
   @param：text：文本内容
   @param：pos：文本位置
   @retval:返回成功输出的字符长度，失败返回-1.
*/
int CvxText::putText(cv::Mat& img, const wchar_t* text, cv::Point pos)
{
    return putText(img, text, pos, CV_RGB(255,255,255));
}


/* @brief: 输出汉字(颜色默认为黑色).遇到不能输出的字符将停止.
   @param：img：输出的影象
   @param：text：文本内容
   @param：pos：文本位置
   @param：color：文本颜色
   @retval:返回成功输出的字符长度，失败返回-1.
*/
int CvxText::putText(cv::Mat& img, const char* text, cv::Point pos, cv::Scalar color)
{
    if (img.data == nullptr) return -1;
    if (text == nullptr) return -1;

    int i;
    for (i = 0; text[i] != '\0'; ++i) {
        wchar_t wc = text[i];

        // 解析双字节符号
        if(!isascii(wc)) mbtowc(&wc, &text[i++], 2);

        // 输出当前的字符
        putWChar(img, wc, pos, color);
    }

    return i;
}


/* @brief: 输出汉字(颜色默认为黑色).遇到不能输出的字符将停止.
   @param：img：输出的影象
   @param：text：文本内容
   @param：pos：文本位置
   @param：color：文本颜色
   @retval:返回成功输出的字符长度，失败返回-1.
*/
int CvxText::putText(cv::Mat& img, const wchar_t* text, cv::Point pos, cv::Scalar color)
{
    if (img.data == nullptr) return -1;
    if (text == nullptr) return -1;

    int i;
    for(i = 0; text[i] != '\0'; ++i) {
        // 输出当前的字符
        putWChar(img, text[i], pos, color);
    }

    return i;
}


/* @brief: 输出当前字符, 更新m_pos位置
   @param：img：输出的影象
   @param：wc：文本内容
   @param：pos：文本位置
   @param：color：文本颜色
   @retval:none
*/
void CvxText::putWChar(cv::Mat& img, wchar_t wc, cv::Point& pos, cv::Scalar color)
{
    // 根据unicode生成字体的二值位图
    FT_UInt glyph_index = FT_Get_Char_Index(m_face, wc);
    FT_Load_Glyph(m_face, glyph_index, FT_LOAD_DEFAULT);
    FT_Render_Glyph(m_face->glyph, FT_RENDER_MODE_MONO);

    FT_GlyphSlot slot = m_face->glyph;

    // 行列数
    int rows = slot->bitmap.rows;
    int cols = slot->bitmap.width;

    for (int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            int off  = i * slot->bitmap.pitch + j/8;

            if (slot->bitmap.buffer[off] & (0xC0 >> (j%8))) {
                int r = pos.y - (rows-1-i);
                int c = pos.x + j;

                if(r >= 0 && r < img.rows && c >= 0 && c < img.cols) {
                    cv::Vec3b pixel = img.at<cv::Vec3b>(cv::Point(c, r));
                    cv::Scalar scalar = cv::Scalar(pixel.val[0], pixel.val[1], pixel.val[2]);

                    // 进行色彩融合
                    float p = m_fontDiaphaneity;
                    for (int k = 0; k < 4; ++k) {
                        scalar.val[k] = scalar.val[k]*(1-p) + color.val[k]*p;
                    }

                    img.at<cv::Vec3b>(cv::Point(c, r))[0] = (unsigned char)(scalar.val[0]);
                    img.at<cv::Vec3b>(cv::Point(c, r))[1] = (unsigned char)(scalar.val[1]);
                    img.at<cv::Vec3b>(cv::Point(c, r))[2] = (unsigned char)(scalar.val[2]);
                }
            }
        }
    }

    // 修改下一个字的输出位置
    double space = m_fontSize.val[0]*m_fontSize.val[1];
    double sep   = m_fontSize.val[0]*m_fontSize.val[2];

    pos.x += (int)((cols? cols: space) + sep);
}


/* @brief: 将char*转为wchar*
   @param：src：
   @param：dest：
   @param：locale：
   @retval:成功返回0，失败返回-1.
*/
int CvxText::ToWchar(char* src, wchar_t* &dest, const char *locale )
{
    if (src == NULL) {
        dest = NULL;
        return 0;
    }

    // 根据环境变量设置locale
    setlocale(LC_CTYPE, locale);

    // 得到转化为需要的宽字符大小
    int w_size = mbstowcs(NULL, src, 0) + 1;

    // w_size = 0 说明mbstowcs返回值为-1。即在运行过程中遇到了非法字符(很有可能使locale
    // 没有设置正确)
    if (w_size == 0) {
        dest = NULL;
        return -1;
    }

    dest = new wchar_t[w_size];
    if (!dest) {
        return -1;
    }

    int ret = mbstowcs(dest, src, strlen(src)+1);
    if (ret <= 0) {
        return -1;
    }
    return 0;
}