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
// confidence thresh for tracking     
const float CONF_THRES = 0.35;
auto uint8Deleter = [] (uint8_t* p) { };
}

namespace MxBase {
    // initialization
    APP_ERROR FairmotPostProcess::Init(const std::map <std::string, std::shared_ptr<void>> &postConfig) {  
        LogDebug << "Start to Init FairmotPostProcess.";    
        LogDebug << "End to Init FairmotPostProcess.";
        return APP_ERR_OK;
    }
    // This interface is called only once to implement de initialization tasks (such as memory release).
    APP_ERROR FairmotPostProcess::DeInit() {          
        return APP_ERR_OK;
    }

    // Judge whether the input from tensorinfer is valid
    bool FairmotPostProcess::IsValidTensors(const std::vector <TensorBase> &tensors) const {
        int fairmotType_ = 4;
        if (tensors.size() != (size_t) fairmotType_) {               
            LogError << "number of tensors (" << tensors.size() << ") " << "is unequal to fairmotType_("
                     << fairmotType_ << ")";
            return false;
        }    
        // shape0 shoud be equal to 1*152*272*128
        // shape1 shoud be equal to 1*152*272*2
        // shape2 shoud be equal to 1*152*272*4
        // shape3 shoud be equal to 1*152*272
        auto shape0 = tensors[0].GetShape();        
        auto shape1 = tensors[1].GetShape();
        auto shape2 = tensors[2].GetShape();
        auto shape3 = tensors[3].GetShape();        
        int s0 = (shape0.size() == 4) && (shape0[0] == 1) && (shape0[1] == 152) && 
            (shape0[2] == 272) && (shape0[3] == 128);
        int s1 = (shape1.size() == 4) && (shape1[0] == 1) && (shape1[1] == 152) && 
            (shape1[2] == 272) && (shape1[3] == 2);
        int s2 = (shape2.size() == 4) && (shape2[0] == 1) && (shape2[1] == 152) && 
            (shape2[2] == 272) && (shape2[3] == 4);
        int s3 = (shape3.size() == 3) && (shape3[0] == 1) && (shape3[1] == 152) && 
            (shape3[2] == 272);
        if(s0 && s1 && s2 && s3 == 1)
        {
            return true;
        }  
        else
        {
            return false;
        }
        return true;
    }
    
/*
 * @description: Post-process the network output and calculate coordinates: bbox_top_left x, y; bbox_bottom_right x, y; conf_score;class (all zeros [only human])
 */
    void FairmotPostProcess::ObjectDetectionOutput(const std::vector <TensorBase> &tensors,
                                                   std::vector <std::vector<ObjectInfo>> &objectInfos,
                                                   const std::vector <ResizedImageInfo> &resizedImageInfos) {
        LogDebug << "FairmotPostProcess start to write results.";
        // Judge whether the input from tensorinfer is empty
        if (tensors.size() == 0) {              
            return;
        }
        auto shape = tensors[0].GetShape();     
        if (shape.size() == 0) {
            return;
        }     

        // @param featLayerData  Vector of 4 output feature data        
        std::vector <std::shared_ptr<void>> featLayerData = {};             
        std::vector <std::vector<size_t>> featLayerShapes = {}; 
        for (uint32_t j = 0; j < tensors.size(); j++) {
            auto dataPtr = (uint8_t *) GetBuffer(tensors[j], 0); 
            std::shared_ptr<void> tmpPointer;
            tmpPointer.reset(dataPtr, uint8Deleter);
            // featLayerData stores the head address of 4 output feature data
            featLayerData.push_back(tmpPointer);                                
            shape = tensors[j].GetShape();

            std::vector <size_t> featLayerShape = {};
            for (auto s : shape) {
                featLayerShape.push_back((size_t) s);
            }
            // featLayerShapes stores the shapes of 4 output feature data
            featLayerShapes.push_back(featLayerShape);             
        }

        // tensors[0] matchs id_feature,id_feature is not used in postprocess
        // tensors[1] matchs reg
        // tensors[2] matchs wh
        // tensors[3] matchs hm
        // Get the head address of hm
        std::shared_ptr<void> hm_addr = featLayerData[3];
        // Create a vector container XY to store coordinate information
        std::vector<std::vector<int>> XY;
        for(uint32_t i = 0;i < 152*272 ; i++){
            // Compared with the threshold CONF_THRES to obtain coordinate information
            if(static_cast<float *>(hm_addr.get())[i] > CONF_THRES)
            {
                std::vector<int>xy;                
                int x = i / 272;
                int y = i - 272 * x;
                xy.push_back(x);
                xy.push_back(y); 
                XY.push_back(xy);  
            }
        }
        // Create a vector container scores to store the information in the corresponding coordinate XY in hm
        std::vector<float>scores;
        for(uint32_t i = 0;i < XY.size();i++){
            scores.push_back(static_cast<float *>(hm_addr.get())[XY[i][0] * 272 + XY[i][1]]);
        }
        // Get the head address of wh and reg
        std::shared_ptr<void> wh_addr = featLayerData[2];
        std::shared_ptr<void> reg_addr = featLayerData[1];
        // std::shared_ptr<void> id_feature_addr = featLayerData[0];

        // WH: n*4
        std::vector<std::vector<float>>WH;
        for(int i = 0; i < XY.size();i++){
            std::vector<float>wh;
            for(int j = 0;j < 4;j++){
                wh.push_back(static_cast<float *>(wh_addr.get())[(XY[i][0] * 272 + XY[i][1]) * 4 + j]);
            }
            WH.push_back(wh);
        }

        // REG: n*2
        std::vector<std::vector<float>>REG;
        for(int i = 0; i < XY.size();i++){
            std::vector<float>reg;
            for(int j = 0;j < 2;j++){
                reg.push_back(static_cast<float *>(reg_addr.get())[(XY[i][0] * 272 + XY[i][1]) * 2 + j]);
            }
            REG.push_back(reg);
        }         

        // XY_f changes the data in XY from int to float
        std::vector<std::vector<float>> XY_f;
        for(int i = 0;i < XY.size();i++){
            std::vector<float>xy_f;
            xy_f.push_back(XY[i][0]);
            xy_f.push_back(XY[i][1]);
            XY_f.push_back(xy_f);
        }
  
        for(int i = 0;i < XY_f.size();i++){                
            XY_f[i][1] = XY_f[i][1] + REG[i][0];
            XY_f[i][0] = XY_f[i][0] + REG[i][1];
        }
        // dets: n*6
        std::vector<std::vector<float>>dets;
        for(int i = 0;i < XY.size();i++){
            std::vector<float>det;
            det.push_back(XY_f[i][1] - WH[i][0]);
            det.push_back(XY_f[i][0] - WH[i][1]);
            det.push_back(XY_f[i][1] + WH[i][2]);
            det.push_back(XY_f[i][0] + WH[i][3]);
            det.push_back(scores[i]);
            det.push_back(0);
            dets.push_back(det);
        }

        // Width and height of initial video
        int width = resizedImageInfos[0].widthOriginal;          
        int height = resizedImageInfos[0].heightOriginal; 
        // Scaled width and height           
        int inp_height = resizedImageInfos[0].heightResize;      
        int inp_width = resizedImageInfos[0].widthResize;   
        // Create a vector container center to store the center point of the original picture
        std::vector<float>c;
        c.push_back(width / 2);
        c.push_back(height / 2);
        std::vector<float>center(c);
         
        float scale = 0;
        scale = std::max(float(inp_width) / float(inp_height) * height, float(width)) * 1.0 ;
        std::vector<float>Scale;
        Scale.push_back(scale);
        Scale.push_back(scale);
        
        int down_ratio = 4;
        int h = inp_height / down_ratio ;
        int w = inp_width / down_ratio ;
        std::vector<int>output_size;
        output_size.push_back(w);
        output_size.push_back(h);

        int rot = 0;

        std::vector<float>shift(2,0);
        
        int inv = 1;

        // get_affine_transform
        std::vector<float>scale_tmp(Scale);
        float src_w = scale_tmp[0];
        int dst_w = output_size[0];
        int dst_h = output_size[1];

        float rot_rad = 3.1415926 * rot / 180;

        std::vector<float>src_point;
        src_point.push_back(0);
        src_point.push_back(src_w * (-0.5));

        float sn = sin(rot_rad);
        float cs = cos(rot_rad);
        // get_dir
        std::vector<float>src_dir(2,0);
        src_dir[0] = src_point[0] * cs - src_point[1] * sn ;
        src_dir[1] = src_point[0] * sn + src_point[1] * cs ;  

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
        
        // get_3rd_point
        std::vector<float>direct;
        direct.push_back(src[0][0]-src[1][0]);
        direct.push_back(src[0][1]-src[1][1]);
        src[2][0] = src[1][0] - direct[1];
        src[2][1] = src[1][1] + direct[0];
        // get_3rd_point
        direct[0] = dst[0][0] - dst[1][0];
        direct[1] = dst[0][1] - dst[1][1];
        dst[2][0] = dst[1][0] - direct[1];
        dst[2][1] = dst[1][1] + direct[0];

        // change data in src and dst to point2f format  
        Point2f SRC[3];
        Point2f DST[3];
        SRC[0] = Point2f(src[0][0],src[0][1]);
        SRC[1] = Point2f(src[1][0],src[1][1]);
        SRC[2] = Point2f(src[2][0],src[2][1]);

        DST[0] = Point2f(dst[0][0],dst[0][1]);
        DST[1] = Point2f(dst[1][0],dst[1][1]);
        DST[2] = Point2f(dst[2][0],dst[2][1]);

        Mat trans(2, 3, CV_64FC1);
        if(inv == 1){
            trans = cv::getAffineTransform(DST,SRC);
        }
        else{
            trans = cv::getAffineTransform(SRC,DST);
        }
        // Get data from mat type trans to array Trans
        float Trans[2][3];
        for(int i = 0;i < 2;i++){
            for(int j = 0;j < 3;j++){
                Trans[i][j] = trans.at<double>(i,j);
            }
        }
        // affine_transform
        for(int i = 0;i < dets.size();i++){
            float new_pt[3] = {dets[i][0], dets[i][1], 1 };
            dets[i][0] = Trans[0][0]* new_pt[0] + Trans[0][1]* new_pt[1] + Trans[0][2]* new_pt[2]; 
            dets[i][1] = Trans[1][0]* new_pt[0] + Trans[1][1]* new_pt[1] + Trans[1][2]* new_pt[2]; 
        }
        for(int i = 0;i < dets.size();i++){
            float new_pt[3] = {dets[i][2], dets[i][3], 1 };
            dets[i][2] = Trans[0][0]* new_pt[0] + Trans[0][1]* new_pt[1] + Trans[0][2]* new_pt[2]; 
            dets[i][3] = Trans[1][0]* new_pt[0] + Trans[1][1]* new_pt[1] + Trans[1][2]* new_pt[2]; 
        }
        // output
        std::vector <ObjectInfo> objectInfo;        
        for(int i = 0;i < dets.size();i++){
            ObjectInfo objInfo;
            objInfo.classId = 0;
            objInfo.confidence = dets[i][4];
            objInfo.className = "human";
            // Normalization
            objInfo.x0 = dets[i][0] / resizedImageInfos[0].widthOriginal;
            objInfo.y0 = dets[i][1] / resizedImageInfos[0].heightOriginal;
            objInfo.x1 = dets[i][2] / resizedImageInfos[0].widthOriginal;
            objInfo.y1 = dets[i][3] / resizedImageInfos[0].heightOriginal;
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