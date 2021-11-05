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
 
#include "lpr_postprocess.h"

using namespace std;

/* @brief: 后处理初始化
   @param：initParam：初始化所需的结构体参数
   @retval:APP_ERROR型变量
*/
APP_ERROR LPRPostProcess::init(const InitParam &initParam)
{
    return APP_ERR_OK;
}


/* @brief: 释放各类资源
   @param：none
   @retval:none
*/
APP_ERROR LPRPostProcess::deinit()
{
    return APP_ERR_OK;
}


/* @brief: 后处理主程序
   @param：recog_outputs：车牌识别模型的推理输出Tensor
   @param：objectInfo：目标检测类任务的信息框变量，将车牌号存入其className成员中
   @retval:none
*/
APP_ERROR LPRPostProcess::process(std::vector<MxBase::TensorBase> recog_outputs, MxBase::ObjectInfo& objectInfo)
{
    // 车牌号码的字符映射关系
    std::map<int, std::string> char_map = {
    {0 ,"京"},{1 ,"沪"},{2 ,"津"},{3 ,"渝"},{4 ,"冀"},{5, "晋"},{6, "蒙"},{7, "辽"},{8, "吉"},{9, "黑"},{10,"苏"},
    {11,"浙"},{12,"皖"},{13,"闽"},{14,"赣"},{15,"鲁"},{16,"豫"},{17,"鄂"},{18,"湘"},{19,"粤"},{20,"桂"},{21,"琼"},
    {22,"川"},{23,"贵"},{24,"云"},{25,"藏"},{26,"陕"},{27,"甘"},{28,"青"},{29,"宁"},{30,"新"},{31,"0"},{32,"1"},
    {33,"2"},{34,"3"},{35,"4"},{36,"5"},{37,"6"},{38,"7"},{39,"8"},{40,"9"},{41,"A"},{42,"B"},{43,"C"},{44,"D"},
    {45,"E"},{46,"F"},{47,"G"},{48,"H"},{49,"J"},{50,"K"},{51,"L"},{52,"M"},{53,"N"},{54,"P"},{55,"Q"},{56,"R"},
    {57,"S"},{58,"T"},{59,"U"},{60,"V"},{61,"W"},{62,"X"},{63,"Y"},{64,"Z"} };

    std::vector<float *> chrs;
    float * chr = nullptr;
    for(int i = 0; i<int(recog_outputs.size()); i++)
    {
        recog_outputs[i].ToHost(); // 将数据从Device侧转移到Host侧，并存入容器
        chr = (float *)recog_outputs[i].GetBuffer();
        chrs.push_back(chr);
    }

    // 以下进行后处理
    float max_val = 0;
    int max_index = 0;
    std::stringstream fmt;
    std::vector<std::string> carplate_chrs = {}; // 存放预测的7个车牌字符

    // 简介：该模型只能预测7个字符的蓝底车牌，其它颜色的车牌效果较差；第一个字符的识别准确率较低；完全正确识别7个字符的准确率较低。
    // 原理：模型输出7个1×65的Tensor，Tensor内的值对应字符映射关系char_map中每个字符的概率，概率最大值所对应的字符即为所预测的字符。
    for(int i = 0; i<int(chrs.size()); i++)
    {
        for(int j = 0; j<65; j++) // 得出概率最大值的索引
        {
            if(chrs[i][j] >= max_val)
            {
                max_val = chrs[i][j];
                max_index = j;
            }
        }
        carplate_chrs.push_back(char_map[max_index]); 
        max_val = 0;
        max_index = 0;
     }

    fmt << carplate_chrs[0] << carplate_chrs[1] << carplate_chrs[2] << carplate_chrs[3] << carplate_chrs[4] 
        << carplate_chrs[5] << carplate_chrs[6];

    objectInfo.className = fmt.str();

    return APP_ERR_OK;
}

