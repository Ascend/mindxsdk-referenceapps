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

#include "FairmotPostProcess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
using namespace cv;

namespace {
// const int SCALE = 32;
// const int BIASESDIM = 2;
// const int OFFSETWIDTH = 2;
// const int OFFSETHEIGHT = 3;
// const int OFFSETBIASES = 1;
// const int OFFSETOBJECTNESS = 1;

// const int NHWC_HEIGHTINDEX = 1;
// const int NHWC_WIDTHINDEX = 2;
// const int NCHW_HEIGHTINDEX = 2;
// const int NCHW_WIDTHINDEX = 3;
// const int YOLO_INFO_DIM = 5;

auto uint8Deleter = [] (uint8_t* p) { };
}
namespace MxBase {
    // FairmotPostProcess &FairmotPostProcess::operator=(const FairmotPostProcess &other) {
    //     if (this == &other) {
    //         return *this;
    //     }
    //     ObjectPostProcessBase::operator=(other);
    //     // objectnessThresh_ = other.objectnessThresh_; // Threshold of objectness value
    //     // iouThresh_ = other.iouThresh_;
    //     // anchorDim_ = other.anchorDim_;
    //     // biasesNum_ = other.biasesNum_;
    //     // yoloType_ = other.yoloType_;
    //     // modelType_ = other.modelType_;
    //     // yoloType_ = other.yoloType_;
    //     // inputType_ = other.inputType_;
    //     // biases_ = other.biases_;
    //     return *this;
    // }

    APP_ERROR FairmotPostProcess::Init(const std::map <std::string, std::shared_ptr<void>> &postConfig) {       //输入postConfig是配置参数
        LogDebug << "Start to Init FairmotPostProcess.";                      //LogDebug 打印调试信息。
        // APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);             //Init 用于完成模型后处理初始化。
        // if (ret != APP_ERR_OK) {
        //     LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
        //     return ret;
        // }

        // configData_.GetFileValue<int>("BIASES_NUM", biasesNum_);
        // std::string str;
        // configData_.GetFileValue<std::string>("BIASES", str);
        // configData_.GetFileValue<float>("OBJECTNESS_THRESH", objectnessThresh_);
        // configData_.GetFileValue<float>("IOU_THRESH", iouThresh_);
        // configData_.GetFileValue<int>("YOLO_TYPE", yoloType_);
        // configData_.GetFileValue<int>("MODEL_TYPE", modelType_);
        // configData_.GetFileValue<int>("YOLO_VERSION", yoloVersion_);
        // configData_.GetFileValue<int>("INPUT_TYPE", inputType_);
        // configData_.GetFileValue<int>("ANCHOR_DIM", anchorDim_);

        // ret = GetBiases(str);
        // if (ret != APP_ERR_OK) {
        //     LogError << GetError(ret) << "Failed to get biases.";
        //     return ret;
        // }
        LogDebug << "End to Init FairmotPostProcess.";
        return APP_ERR_OK;
    }

    APP_ERROR FairmotPostProcess::DeInit() {          // 该接口仅调用一次，用于实现去初始化任务（例如内存释放）。
        return APP_ERR_OK;
    }

    // 判断tensor输出是否有效
    bool FairmotPostProcess::IsValidTensors(const std::vector <TensorBase> &tensors) const {
        int fairmotType_ = 4;
        if (tensors.size() != (size_t) fairmotType_) {               
            LogError << "number of tensors (" << tensors.size() << ") " << "is unequal to fairmotType_("
                     << fairmotType_ << ")";
            return false;
        }    
        auto shape0 = tensors[0].GetShape();
        auto shape1 = tensors[1].GetShape();
        auto shape2 = tensors[2].GetShape();
        auto shape3 = tensors[3].GetShape();        
        int s0 = (shape0.size()==4)&&(shape0[0]==1)&&(shape0[1]==152)&&(shape0[2]==272)&&(shape0[3]==128);
        int s1 = (shape1.size()==4)&&(shape1[0]==1)&&(shape1[1]==152)&&(shape1[2]==272)&&(shape1[3]==2);
        int s2 = (shape2.size()==4)&&(shape2[0]==1)&&(shape2[1]==152)&&(shape2[2]==272)&&(shape2[3]==4);
        int s3 = (shape3.size()==3)&&(shape3[0]==1)&&(shape3[1]==152)&&(shape3[2]==272);
        if(s0&&s1&&s2&&s3 == 1)
        {
            return true;
        }  
        else
        {
            return false;
        }
        return true;
    }

    void FairmotPostProcess::ObjectDetectionOutput(const std::vector <TensorBase> &tensors,
                                                  std::vector <std::vector<ObjectInfo>> &objectInfos,
                                                  const std::vector <ResizedImageInfo> &resizedImageInfos) {
        LogDebug << "FairmotPostProcess start to write results.";   //
        if (tensors.size() == 0) {              //仍在判断tensor是否有效
            return;
        }
        auto shape = tensors[0].GetShape();     //仍在判断tensor是否有效
        if (shape.size() == 0) {
            return;
        }       
        uint32_t batchSize = shape[0];

        
        std::vector <std::shared_ptr<void>> featLayerData = {};             // @param featLayerData  Vector of 3 output feature data

        std::vector <std::vector<size_t>> featLayerShapes = {};             // featLayerShapes

        for (uint32_t j = 0; j < tensors.size(); j++) {                     // tensors.size() = 4

            //std::cout<<" tensors[j] = "<<tensors[j]<<std::endl;

            auto dataPtr = (uint8_t *) GetBuffer(tensors[j], 0);            // dataPtr打印出来是空的

            std::shared_ptr<void> tmpPointer;

            tmpPointer.reset(dataPtr, uint8Deleter);

            // std::cout<<" tmpPointer = "<<tmpPointer <<std::endl;


            featLayerData.push_back(tmpPointer);                    //featLayerData存储首地址指针

            // std::cout<<" featLayerData["<<j<<"] = "<<featLayerData[j] <<std::endl;                 
            
            shape = tensors[j].GetShape();

            std::vector <size_t> featLayerShape = {};
            for (auto s : shape) {
                featLayerShape.push_back((size_t) s);
            }
            featLayerShapes.push_back(featLayerShape);             //featLayerShapes存储形状

            // std::cout<<"featLayerShape = "<< featLayerShape.size() <<std::endl;

            // std::cout<<"featLayerShapes.size() = "<< featLayerShapes[0][0] <<std::endl;

        }




        
        std::shared_ptr<void> hm_addr = featLayerData[3];
        //std::shared_ptr<void> netout = featLayerData[i];
        // std::cout<<"hm_addr = "<< hm_addr <<std::endl;

        // for (uint32_t j = 0; j < tensors.size(); j++) {
        //     std::cout<<" featLayerData["<<j<<"] = "<<featLayerData[j] <<std::endl; 
        // } 

        std::vector<std::vector<int>> XY;

        float conf_thres = 0.35 ;

        for(uint32_t i = 0;i < 152*272 ; i++ ){
            if( static_cast<float *>(hm_addr.get())[i] > conf_thres )
            {
                //std::cout<<  "=="<<static_cast<float *>(hm_addr.get())[i]<<std::endl;

                std::vector<int>xy;
                
                int x = i / 272;
                int y = i - 272 * x;

                xy.push_back(x);
                xy.push_back(y); 
                XY.push_back(xy);  //[ ys,xs ]
            }
            //a = a + 1;
        }


        std::vector<float>scores;
        // std::cout<<"scores.size() = "<<scores.size()<<std::endl;
        // std::cout<<"XY.size() = "<<XY.size()<<std::endl;
        
        // uint32_t temp;
        for(uint32_t i = 0;i < XY.size();i++ ){
            scores.push_back( static_cast<float *>(hm_addr.get())[ XY[i][0]*272 + XY[i][1] ] );
        }

        //tensors[0]对应id_feature
        //tensors[1]对应reg
        //tensors[2]对应wh
        //tensors[3]对应hm
        std::shared_ptr<void> wh_addr = featLayerData[2];
        // std::cout<<"wh_addr = "<< wh_addr <<std::endl;

        std::shared_ptr<void> reg_addr = featLayerData[1];
        // std::cout<<"reg_addr = "<< reg_addr <<std::endl;

        std::shared_ptr<void> id_feature_addr = featLayerData[0];

        //WH: n*4
        std::vector<std::vector<float>>WH;
        for(int i = 0; i < XY.size();i++){
            std::vector<float>wh;
            for(int j=0;j<4;j++){
                wh.push_back( static_cast<float *>(wh_addr.get())[ (XY[i][0]*272 + XY[i][1])*4 + j] );
            }

            WH.push_back(wh);
        }

        // REG: n*2
        std::vector<std::vector<float>>REG;
        for(int i = 0; i < XY.size();i++){
            std::vector<float>reg;
            for(int j=0;j<2;j++){
                reg.push_back( static_cast<float *>(reg_addr.get())[ (XY[i][0]*272 + XY[i][1])*2 + j ] );
            }
            REG.push_back(reg);
        }

        //ID_feature: n*128
        std::vector<std::vector<float>>ID_feature;
        for(int i = 0; i < XY.size();i++){
            std::vector<float>id_feature;
            for(int j=0;j<128;j++){
                id_feature.push_back( static_cast<float *>(id_feature_addr.get())[ (XY[i][0]*272 + XY[i][1])*128 + j ] );
            }
            ID_feature.push_back(id_feature);
        }            

        //XY_f拷贝XY并将数据类型从整形变成浮点型
        std::vector<std::vector<float>>XY_f;
        for(int i =0;i<XY.size();i++){
            std::vector<float>xy_f;
            xy_f.push_back(XY[i][0]);
            xy_f.push_back(XY[i][1]);
            XY_f.push_back(xy_f);
        }

        //XY = [ys , xs] ;  XY_f = [ys , xs]     
        for(int i=0;i<XY_f.size();i++){                
            XY_f[i][1] = XY_f[i][1] + REG[i][0];
            XY_f[i][0] = XY_f[i][0] + REG[i][1];
        }
        
        std::vector<std::vector<float>>dets;
        for(int i = 0;i<XY.size();i++){
            std::vector<float>det;
            det.push_back(XY_f[i][1] - WH[i][0]);
            det.push_back(XY_f[i][0] - WH[i][1]);
            det.push_back(XY_f[i][1] + WH[i][2]);
            det.push_back(XY_f[i][0] + WH[i][3]);
            det.push_back(scores[i]);
            det.push_back(0);
            dets.push_back(det);
        }


        int width = resizedImageInfos[0].widthOriginal;          //原图大小
        int height = resizedImageInfos[0].heightOriginal;             
        int inp_height = resizedImageInfos[0].heightResize;      //缩放之后的大小
        int inp_width = resizedImageInfos[0].widthResize;   

        int down_ratio = 4;

        std::vector<float>c;
        c.push_back(width/2);
        c.push_back(height/2);

        float s;
        s = std::max(float(inp_width) / float(inp_height) * height, float(width)) * 1.0 ;

        int h = inp_height / down_ratio ;
        int w = inp_width / down_ratio ;
            
        int num_classes = 1;

        // for(int i = 0; i < 1; i++){
            
        std::vector<float>center(c);
        float scale = s;
        int rot = 0;
        std::vector<int>output_size;
        output_size.push_back(w);
        output_size.push_back(h);
        std::vector<float>shift(2,0);

        
        int inv = 1;
        
        std::vector<float>Scale;
        Scale.push_back(scale);
        Scale.push_back(scale);

        std::vector<float>scale_tmp(Scale);

        float src_w = scale_tmp[0];
        int dst_w = output_size[0];
        int dst_h = output_size[1];

        float rot_rad = 0;

        std::vector<float>src_point;
        src_point.push_back(0);
        src_point.push_back(src_w * (-0.5));

        float sn = 0;
        float cs = 1;

        std::vector<float>src_result(2,0);
        src_result[0] = src_point[0] * cs - src_point[1] * sn ;
        src_result[1] = src_point[0] * sn + src_point[1] * cs ;  
        std::vector<float>src_dir(src_result);

        std::vector<float>dst_dir;
        dst_dir.push_back(0);
        dst_dir.push_back(dst_w * (-0.5));

        float src[3][2] = {0};
        float dst[3][2] = {0};

        src[0][0] = center[0];
        src[0][1] = center[1];
        src[1][0] = center[0] + src_dir[0];
        src[1][1] = center[1] + src_dir[1];
        dst[0][0] = dst_w * 0.5;
        dst[0][1] = dst_h * 0.5;
        dst[1][0] = dst_w * 0.5 + dst_dir[0];
        dst[1][1] = dst_h * 0.5 + dst_dir[1];

        std::vector<float>direct;
        direct.push_back(src[0][0]-src[1][0]);
        direct.push_back(src[0][1]-src[1][1]);

        src[2][0] = src[1][0] - direct[1];
        src[2][1] = src[1][1] + direct[0];

        direct[0] = dst[0][0] - dst[1][0];
        direct[1] = dst[0][1] - dst[1][1];

        dst[2][0] = dst[1][0] - direct[1];
        dst[2][1] = dst[1][1] + direct[0];

        //转格式
        Point2f SRC[3];
        Point2f DST[3];
        SRC[0] = Point2f(src[0][0],src[0][1]);
        SRC[1] = Point2f(src[1][0],src[1][1]);
        SRC[2] = Point2f(src[2][0],src[2][1]);

        DST[0] = Point2f(dst[0][0],dst[0][1]);
        DST[1] = Point2f(dst[1][0],dst[1][1]);
        DST[2] = Point2f(dst[2][0],dst[2][1]);



        Mat trans( 2, 3, CV_64FC1 );

        trans = cv::getAffineTransform( DST , SRC);


        //从MAT型数据trans中把数据拿到矩阵Trans中
        float Trans[2][3];
        for(int i=0;i<2;i++){
            for(int j=0;j<3;j++){
                Trans[i][j] = trans.at<double>(i,j);
            }
        }

        std::vector<std::vector<float>>target_coords; 
        for(int i =0;i<dets.size();i++){
            std::vector<float>target(2,0);
            target_coords.push_back(target);
        }
        
        for(int i = 0;i<dets.size();i++){
            float new_pt[3] = {dets[i][0], dets[i][1], 1 };
            dets[i][0] = Trans[0][0]* new_pt[0] + Trans[0][1]* new_pt[1] + Trans[0][2]* new_pt[2]; 
            dets[i][1] = Trans[1][0]* new_pt[0] + Trans[1][1]* new_pt[1] + Trans[1][2]* new_pt[2]; 
        }
        for(int i = 0;i<dets.size();i++){
            float new_pt[3] = {dets[i][2], dets[i][3], 1 };
            dets[i][2] = Trans[0][0]* new_pt[0] + Trans[0][1]* new_pt[1] + Trans[0][2]* new_pt[2]; 
            dets[i][3] = Trans[1][0]* new_pt[0] + Trans[1][1]* new_pt[1] + Trans[1][2]* new_pt[2]; 
        }
            
    
        // }

        std::vector <ObjectInfo> objectInfo;        
        for(int i =0;i<dets.size();i++){
            ObjectInfo objInfo;
            objInfo.classId = 0;
            objInfo.confidence = dets[i][4];
            objInfo.className = " ";
            objInfo.x0 = dets[i][0]/resizedImageInfos[0].widthOriginal;
            objInfo.y0 = dets[i][1]/resizedImageInfos[0].heightOriginal;
            objInfo.x1 = dets[i][2]/resizedImageInfos[0].widthOriginal;
            objInfo.y1 = dets[i][3]/resizedImageInfos[0].heightOriginal;
            objectInfo.push_back(objInfo);
        }

        objectInfos.push_back(objectInfo);
        
        LogDebug << "FairmotPostProcess write results successed.";   
    }







    APP_ERROR FairmotPostProcess::Process(const std::vector <TensorBase> &tensors,
                                         std::vector <std::vector<ObjectInfo>> &objectInfos,
                                         const std::vector <ResizedImageInfo> &resizedImageInfos,
                                         const std::map <std::string, std::shared_ptr<void>> &paramMap) {
        LogDebug << "Start to Process FairmotPostProcess.";
        APP_ERROR ret = APP_ERR_OK;
        if (resizedImageInfos.size() == 0) {
            ret = APP_ERR_INPUT_NOT_MATCH;
            LogError << GetError(ret) << "resizedImageInfos is not provided which is necessary for FairmotPostProcess.";
            return ret;
        }
        auto inputs = tensors;
        ret = CheckAndMoveTensors(inputs);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "CheckAndMoveTensors failed.";
            return ret;
        }
        ObjectDetectionOutput(inputs, objectInfos,  resizedImageInfos);
        for (uint32_t i = 0; i < resizedImageInfos.size(); i++) {
            CoordinatesReduction(i, resizedImageInfos[i], objectInfos[i]);
        }
        LogObjectInfos(objectInfos);
        LogDebug << "End to Process FairmotPostProcess.";
        return APP_ERR_OK;
    }

    extern "C" {
    std::shared_ptr <MxBase::FairmotPostProcess> GetObjectInstance() {
        LogInfo << "Begin to get FairmotPostProcess instance.";
        auto instance = std::make_shared<MxBase::FairmotPostProcess>();
        LogInfo << "End to get FairmotPostProcess instance.";
        return instance;
    }
    }
}