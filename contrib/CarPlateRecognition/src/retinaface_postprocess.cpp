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

#include <algorithm>
#include <math.h>
#include "retinaface_postprocess.h"


/* @brief:初始化后处理所需的各项参数
   @param:initParam：后处理所需参数的结构体变量
   @retval:APP_ERROR型变量
*/
APP_ERROR RetinaFace_PostProcess::Init(const InitParam &initParam)
{
    SetDefaultParams();

    nmsThreshold_ = initParam.nmsThreshold;
    scoreThreshold_ = initParam.scoreThreshold;
    width_ = initParam.width;
    height_= initParam.height;
    steps_ = initParam.steps;
    min_sizes_ = initParam.min_sizes;
    variances_ = initParam.variances;
    scale_ = initParam.scale;

    return APP_ERR_OK;
}


/* @brief:释放资源
   @param:none
   @retval:APP_ERROR型变量
*/
APP_ERROR RetinaFace_PostProcess::DeInit()
{
    return APP_ERR_OK;
}


/* @brief:将后处理参数设置为默认值
   @param:none
   @retval:none
*/
void RetinaFace_PostProcess::SetDefaultParams(){
    nmsThreshold_ = 0.4;
    scoreThreshold_ = 0.6;
    width_ = 640;
    height_= 640;
    steps_= {8, 16, 32};
    min_sizes_ = {{24, 48}, {96, 192}, {384, 768}};
    variances_ = {0.1, 0.2};
    scale_ = {640, 640, 640, 640};
}


/* @brief:生成锚框anchor
   @param:anchor-自定义的box型容器，用于存放所生成的anchor
   @param:w-图像的宽(经resize后)
   @param:h-图像的高(经resize后)
   @retval:none
   @notice:图像输入前会被resize成640×640
*/
void RetinaFace_PostProcess::GenerateAnchor(std::vector<box> &anchor, int w, int h)
{
    anchor.clear();

    // feature_map= [[80,80], [40,40], [20,20]]
    // 计算原理：step=8时， 640÷8=80，即将输入的图像划分为 80×80个方格
    //          step=16时，640÷8=40；
    //          step=32时，640÷8=20
    std::vector<std::vector<int> > feature_map(3);   
    for (int i = 0; i < int(feature_map.size()); ++i) {
        feature_map[i].push_back(ceil(h/steps_[i])); //ceil是向上取整函数
        feature_map[i].push_back(ceil(w/steps_[i]));
    }

    // 生成锚框anchor
    for (int k = 0; k < int(feature_map.size()); ++k)
    {
        std::vector<int> min_size = min_sizes_[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < int(min_size.size()); ++l)
                {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps_[k]/w; // 计算feature_map中每个方格的中心点的x坐标
                    float cy = (i + 0.5) * steps_[k]/h; // 计算feature_map中每个方格的中心点的y坐标
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }
    }
}


/* @brief:提供给sort函数，用于将容器内的ObjectInfo元素按置信度confidence的大小降序排列
   @param:a：ObjectInfo型变量
   @param:b：ObjectInfo型变量
   @retval:bool
*/
inline bool RetinaFace_PostProcess::cmp(MxBase::ObjectInfo a, MxBase::ObjectInfo b) {
    if (a.confidence > b.confidence)
        return true;
    return false;
}


/* @brief:进行非极大值抑制(Non-Maximum Suprression)
   @param:input_boxes-按照置信度confidence降序排列后的ObjectInfo型容器
   @param:NMS_THRESH-nms阈值
   @retval:none
*/
void RetinaFace_PostProcess::nms(std::vector<MxBase::ObjectInfo> &input_boxes, float NMS_THRESH)
{
    // 计算每个boundingbox框(即bbox)的面积
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x1 - input_boxes.at(i).x0 + 1) // input_boxes.at(i)等价于input_boxes[i]
                   * (input_boxes.at(i).y1 - input_boxes.at(i).y0 + 1);
    }

    // 计算IOU，根据nms阈值进行筛选
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            // 计算两个框重叠区域的面积inter(相当于交集部分面积)
            float xx0 = std::max(input_boxes[i].x0, input_boxes[j].x0); // 计算两个bbox框重叠的矩形区域的左上角的x坐标
            float yy0 = std::max(input_boxes[i].y0, input_boxes[j].y0); // 计算两个bbox框重叠的矩形区域的左上角的y坐标
            float xx1 = std::min(input_boxes[i].x1, input_boxes[j].x1); // 计算两个bbox框重叠的矩形区域的右上角的x坐标
            float yy1 = std::min(input_boxes[i].y1, input_boxes[j].y1); // 计算两个bbox框重叠的矩形区域的右上角的y坐标
            float w = std::max(float(0), xx1 - xx0 + 1); // 计算两个bbox框重叠的矩形区域的宽
            float h = std::max(float(0), yy1 - yy0 + 1); // 计算两个bbox框重叠的矩形区域的高
            float inter = w * h; 

            // 计算IOU，即ovr
            // 其中(vArea[i]+vArea[j]-inter)是两个bbox框的并集部分面积
            float ovr = inter / (vArea[i] + vArea[j] - inter); //

            // 将大于nms阈值的bbox框剔除
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
                j++;
        }
    }
}


/* @brief:后处理主处理函数
   @param:detect_outputs-车牌检测模型的推理输出Tensor
   @param:objectInfos-存放bounding box数据
   @param:resizedImageInfo：图像的缩放方式，用于坐标还原
   @retval:none
*/
APP_ERROR RetinaFace_PostProcess::Process(std::vector<MxBase::TensorBase> detect_outputs, std::vector<MxBase::ObjectInfo>& objectInfos, const MxBase::ResizedImageInfo resizedImageInfo)
{
     // 将数据从Device侧转移到Host侧
    detect_outputs[0].ToHost();
    detect_outputs[1].ToHost();
    detect_outputs[2].ToHost();

    float *landms = (float *)detect_outputs[0].GetBuffer(); // 特征点(该维度没用到)
    float *conf = (float *)detect_outputs[1].GetBuffer();   // 置信度
    float *loc  = (float *)detect_outputs[2].GetBuffer();   // 坐标

    // 生成锚框anchor
    std::vector<box> anchor;
    GenerateAnchor(anchor, width_, height_);

    // 对模型的输出数据进行解码，获取loc conf landms
    std::vector<MxBase::ObjectInfo> total_boxs;
    for (int i = 0; i < int(anchor.size()); ++i)
    {
        if (*(conf+1) > scoreThreshold_) // 进行阈值筛选
        {
            box tmp = anchor[i];
            box tmp1;
            MxBase::ObjectInfo result;

            // 利用anchor对模型输出进行解码，得到loc
            tmp1.cx = tmp.cx + *loc * variances_[0] * tmp.sx;
            tmp1.cy = tmp.cy + *(loc+1) * variances_[0] * tmp.sy;
            tmp1.sx = tmp.sx * exp(*(loc+2) * variances_[1]);
            tmp1.sy = tmp.sy * exp(*(loc+3) * variances_[1]);
            result.x0 = (tmp1.cx - tmp1.sx/2) * scale_[0];
            result.y0 = (tmp1.cy - tmp1.sy/2) * scale_[1];
            result.x1 = (tmp1.cx + tmp1.sx/2) * scale_[2];
            result.y1 = (tmp1.cy + tmp1.sy/2) * scale_[3];

            // score
            result.confidence = *(conf + 1);

            // className
            result.className = "carplate"; // 在车牌识别模型的后处理中，成员className将被用于存放车牌号码

            // classId
            result.classId = 0; // 只能识别一类物体，即车牌，默认将classId置0

            total_boxs.push_back(result);
        }
        loc += 4; // loc代表坐标点，其shape为16800*4，但因为ptr相当于将loc展平为1*67200，所以当某个loc低于阈值时，直接跳过剩下的四个数，
        conf+= 2; // conf代表置信度，其shape为16800*2
        landms += 8; // landms代表特征点，其shape为16800*8，但该维度没用到
    }

    std::sort(total_boxs.begin(), total_boxs.end(), cmp); // 将total_boxs中的元素按置信度confidence大小降序排列
    nms(total_boxs, nmsThreshold_); // 进行非极大值抑制

    for (int j = 0; j < int(total_boxs.size()); ++j)
        objectInfos.push_back(total_boxs[j]);

    // 根据提供的图像缩放方式进行坐标还原
    CoordinatesReduction(0, resizedImageInfo, objectInfos, false);

    return APP_ERR_OK;
}
