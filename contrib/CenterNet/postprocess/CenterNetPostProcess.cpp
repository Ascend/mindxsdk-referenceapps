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

#include "CenterNetPostProcess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"
#include <math.h>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <algorithm>

namespace {
    auto g_uint8Deleter = [] (uint8_t *p) { };
    const int height = 128;
    const int width = 128;
    const int cat = 80;
    const int K = 100;
}

namespace MxBase {
    CenterNetPostProcess::CenterNetPostProcess(const CenterNetPostProcess &other) {
        classNum_ = other.classNum_;
        scoreThresh_ = other.scoreThresh_;
    }

    CenterNetPostProcess &CenterNetPostProcess::operator=(const CenterNetPostProcess &other) {
        if (this == &other) {
            return *this;
        }
        ObjectPostProcessBase::operator=(other);
        classNum_ = other.classNum_;
        scoreThresh_ = other.scoreThresh_;

        return *this;
    }

    APP_ERROR CenterNetPostProcess::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) {
        LogInfo << "Start to Init CenterNetPostProcess.";
        APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
            return ret;
        }
        configData_.GetFileValue<int>("CLASS_NUM", classNum_);
        configData_.GetFileValue<float>("SCORE_THRESH", scoreThresh_);
        LogInfo << "End to Init CenterNetPostProcess.";
        return APP_ERR_OK;
    }

    APP_ERROR CenterNetPostProcess::DeInit() {
        return APP_ERR_OK;
    }

    /**
     * @brief Parsing MxBase::TensorBase data to regression heatmap and classification heatmap of inference model
     * @param tensors - MxBase::TensorBase vector, regression tensor and classification tensor output from the model
     * inference plugin
     * @param regression - Regression heatmap with parsed data, with shape: [batchsize, boxes_num, (dy, dx, dh, dw)]
     * @param classification - Classification heatmap with parsed data
     * */
    void CenterNetPostProcess::ReadDataFromTensor(const std::vector <MxBase::TensorBase> &tensors,
                                                  nc::NdArray<float> &heatmap,
                                                  nc::NdArray<float> &wh,
                                                  nc::NdArray<float> &regression) {
        // Read regression data
        auto hmDataPtr = (uint8_t *)tensors[0].GetBuffer();
        std::shared_ptr<void> hmPointer;
        hmPointer.reset(hmDataPtr, g_uint8Deleter);
        int idx = 0;
        float max0 = 0;
        for (int i = 0; i < cat; i++) {
            for (int j = 0; j < height * width; j++) {
                heatmap(i, j) = fastmath::sigmoid(static_cast<float *>(hmPointer.get())[idx]);
                if(i == 0){
                    if (max0 < heatmap(i, j)){
                        max0 = heatmap(i, j);
                    }
                }
                idx += 1;
            }
        }
        idx = 0;
        cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        for (int i = 0; i < cat; i++) {
            cv::Mat maxpool;
            float load[128][128];
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    load[j][k] = fastmath::sigmoid(static_cast<float *>(hmPointer.get())[idx]);
                    idx += 1;
                }
            }

            cv::Mat tmp(128, 128, CV_32FC1, load);
            dilate(tmp, maxpool, element);
            std::vector<float> tmpv = maxpool.reshape(1, 1);
            for (int j = 0; j < height * width; j++){
                if(heatmap(i, j) != tmpv[j]) {
                    heatmap(i, j) = 0;
                }
            }
        }
        // Read wh data
        auto whDataPtr = (uint8_t *)tensors[1].GetBuffer();
        std::shared_ptr<void> whPointer;
        whPointer.reset(whDataPtr, g_uint8Deleter);
        idx = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < width * height; j++) {
                wh(i, j) = (static_cast<float *>(whPointer.get())[idx]);
                idx += 1;
            }
        }

        auto regDataPtr = (uint8_t *)tensors[2].GetBuffer();
        std::shared_ptr<void> regPointer;
        regPointer.reset(regDataPtr, g_uint8Deleter);
        idx = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < width * height; j++) {
                regression(i, j) = (static_cast<float *>(regPointer.get())[idx]);
                idx += 1;
            }
        }
    }


    nc::NdArray<float> CenterNetPostProcess::_gather_feat(nc::NdArray<float> feat, nc::NdArray<uint32_t> ind) {
        nc::NdArray<float> result;
        int dim  = feat.shape().cols;//第二维
        ind.reshape(ind.shape().cols, ind.shape().rows);//unsqueeze(2)
        if (2 == dim)
        {
            ind = nc::concatenate({ind, ind},nc::Axis::COL);//expand(ind.size(0), ind.size(1), dim)#expand扩展某个size为1的维度到dim维,[100, 1]=>[100,dim]
            result = nc::zeros<float>(K, 2);

            for (int i = 0; i < K; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    result(i, j) = feat(ind(i, j), j);   // gather
                }
            }
        }
        else
        {
            result = nc::zeros<float>(K, 1);
            for (int i = 0; i < K; i++)
            {
                result(i, 0) = feat(ind(i, 0) ,0);   //gather
            }
        }
        return result; // [100, dim]
    }
    nc::NdArray<uint32_t> CenterNetPostProcess::_gather_feat(nc::NdArray<uint32_t> feat, nc::NdArray<uint32_t> ind) {
        nc::NdArray<uint32_t> result;
        int dim = feat.shape().cols;//第二维
        ind.reshape(ind.shape().cols, ind.shape().rows);//unsqueeze(2)
        if (2 == dim)
        {
            ind = nc::concatenate({ind, ind}, nc::Axis::COL);//expand(ind.size(0), ind.size(1), dim)#expand扩展某个size为1的维度到dim维,[1,500,1]=>[1,500,dim]
            result = nc::zeros<uint32_t>(K, 2);

            for (int i = 0; i < K; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    result(i,j) = feat(ind(i, j), j);   //gather
                }
            }
        }
        else
        {
            result = nc::zeros<uint32_t>(K, 1);
            for (int i = 0; i < K; i++)
            {
                result(i,0) = feat(ind(i,0),0);   //gather
            }
        }
        return result;
    }


    void CenterNetPostProcess::_tranpose_and_gather_feat(nc::NdArray<float> &feat, nc::NdArray<uint32_t> ind)//[dim,128*128], [100, dim]
    {
        nc::NdArray<float> feat_trans = feat.transpose();
        feat = _gather_feat(feat_trans, ind);
    }


    void CenterNetPostProcess::_topk(nc::NdArray<float> heat, int K, nc::NdArray<float> &topk_score,
                                     nc::NdArray<uint32_t> &topk_inds, nc::NdArray<uint32_t> &topk_clses,
                                     nc::NdArray<uint32_t> &topk_ys, nc::NdArray<uint32_t> &topk_xs) {
        nc::NdArray<float> topk_scores = nc::zeros<float>(cat, K);
        nc::NdArray<uint32_t> topk_inds_3d = nc::zeros<uint32_t>(cat, K);
        naive_arg_topK_3d(heat, K, topk_scores, topk_inds_3d);

        for (int i = 0; i < cat; i++)
        {
            for (int j = 0; j < K; j++)
            {
                topk_inds_3d(i, j) = topk_inds_3d(i, j)%(height * width);
                topk_ys(i,j) = (topk_inds_3d(i, j) / width);
                topk_xs(i,j) = (topk_inds_3d(i, j) % width);
            }
        }
        nc::NdArray<uint32_t> topk_ind = nc::zeros<uint32_t>(1, K);
        naive_arg_topK_2d(topk_scores.reshape(1, cat * K), K, topk_score, topk_ind);

        for (int j = 0; j < K; j++)
        {
            topk_clses(0, j) = (topk_ind(0, j) / K);
        }

        nc::NdArray<uint32_t> topk_inds_ = _gather_feat(topk_inds_3d.reshape(cat*K, 1), topk_ind);
        nc::NdArray<uint32_t> topk_ys_ = _gather_feat(topk_ys.reshape(cat*K, 1), topk_ind);
        nc::NdArray<uint32_t> topk_xs_ = _gather_feat(topk_xs.reshape(cat*K, 1), topk_ind);
        topk_inds = nc::copy(topk_inds_.reshape(1, K));
        topk_ys = nc::copy(topk_ys_.reshape(1, K));
        topk_xs = nc::copy(topk_xs_.reshape(1, K));

        //return topk_score, topk_inds, topk_clses, topk_ys, topk_xs;
    }


    nc::NdArray<float> CenterNetPostProcess::get_3rd_point(nc::NdArray<float> a, nc::NdArray<float> b){
        nc::NdArray<float> direct = a - b;
        return b + nc::NdArray<float>{-direct[1], direct[0]};
}

    nc::NdArray<float> CenterNetPostProcess::get_dir(nc::NdArray<float> src_point, float rot_rad) {
        float sn = nc::sin(rot_rad);
        float cs = nc::cos(rot_rad);
        nc::NdArray<float> src_result = {0, 0};
        src_result[0] = src_point[0] * cs - src_point[1] * sn;
        src_result[1] = src_point[0] * sn + src_point[1] * cs;

        return src_result;
    }


    cv::Mat CenterNetPostProcess::get_affine_transform(nc::NdArray<float> center, float scale, float rot,
                                                       int output_size[2],nc::NdArray<float> shift = {0, 0}, int inv=0){
        nc::NdArray<float> scales = {scale, scale};//创建矩阵
        nc::NdArray<float> scale_tmp = scales;
        float src_w = scale_tmp[0];
        int dst_w = output_size[0];
        int dst_h = output_size[1];

        float rot_rad = nc::constants::pi * rot / 180;
        nc::NdArray<float> src_dir = get_dir(nc::NdArray<float>{0, float(src_w * -0.5)}, rot_rad);
        nc::NdArray<float> dst_dir = {0, float(dst_w * -0.5)};

        nc::NdArray<float> src = nc::zeros<float>(3, 2);
        nc::NdArray<float> dst = nc::zeros<float>(3, 2);

        nc::NdArray<float> temp;
        temp = center + scale_tmp * shift;
        src(0, 0) = temp(0,0);
        src(0, 1) = temp(0,1);
        temp = center + src_dir + scale_tmp * shift;
        src(1, 0) = temp(0,0);
        src(1, 1) = temp(0,1);
        temp = dst_w * 0.5, dst_h * 0.5;
        dst(0, 0) = temp(0,0);
        dst(0, 1) = temp(0,1);
        temp = nc::NdArray<float>{float(dst_w * 0.5), float(dst_h * 0.5)} + dst_dir;
        dst(1, 0) = temp(0,0);
        dst(1, 1) = temp(0,1);
        temp = get_3rd_point(src(0, src.cSlice()), src(1, src.cSlice()));
        src(2, 0) = temp(0,0);
        src(2, 1) = temp(0,1);
        temp = get_3rd_point(dst(0, dst.cSlice()), dst(1, dst.cSlice()));
        dst(2, 0) = temp(0,0);
        dst(2, 1) = temp(0,1);

        cv::Mat trans;
        cv::Point2f SRC[3];
        cv::Point2f DST[3];

        // Set your 3 points to calculate the  Affine Transform
        SRC[0] = cv::Point2f(src(0,0),src(0,1));//numcpp to opencv`s point2f
        SRC[1] = cv::Point2f(src(1,0),src(1,1));
        SRC[2] = cv::Point2f(src(2,0),src(2,1));

        DST[0] = cv::Point2f(dst(0,0),dst(0,1));
        DST[1] = cv::Point2f(dst(1,0),dst(1,1));
        DST[2] = cv::Point2f(dst(2,0),dst(2,1));

        if (1 == inv)
        {
            trans = cv::getAffineTransform(DST, SRC);
        }
        else
        {
            trans = cv::getAffineTransform(SRC, DST);
        }
        return trans;
}


    nc::NdArray<float> CenterNetPostProcess::affine_transform(nc::NdArray<float> pt, nc::NdArray<float> t){
        nc::NdArray<float> new_pt = {pt(0,0), pt(0,1),1.0};
        new_pt = new_pt.transpose();
        nc::NdArray<float> new_pt_dot = nc::dot(t, new_pt);
        return new_pt_dot({0,2},0);
    }


    void CenterNetPostProcess::naive_arg_topK_3d(nc::NdArray<float> matrix, int K, nc::NdArray<float> &max_score,
                                                 nc::NdArray<uint32_t> &max_k) {
        nc::NdArray<uint32_t> full_sort = nc::argsort(-matrix, nc::Axis::COL); // "-"是降序排列,full_sort是索引,nc::Axis=0竖向，1行向
        max_k = nc::copy(full_sort(full_sort.rSlice(), {0, K})); // 取前K个数, 深拷贝：full_sort和max_k的整体是一个独立的对象。
        for (int i = 0; i < cat; i++) {
            for (int j = 0; j < K; j++) {
                max_score(i, j) = matrix(i, max_k(i, j));
            }
        }
        //return max_score,max_k
    }

    void CenterNetPostProcess::naive_arg_topK_2d(nc::NdArray<float> matrix, int K, nc::NdArray<float> &max_score, nc::NdArray<uint32_t> &max_k) {
        nc::NdArray<uint32_t> full_sort = nc::argsort(-matrix, nc::Axis::COL); //"-"是降序排列,full_sort是索引,nc::Axis=0竖向，1行向
        max_k = nc::copy(full_sort(full_sort.rSlice(),{0, K})); //取前K个数,nc::Axis=0竖向，1行向# 深拷贝：full_sort和max_k的整体是一个独立的对象。
        //nc::NdArray<float> max_score = nc::zeros<float>(5, K);
        for (int j = 0; j < K; j++)
        {
            max_score(0, j) = matrix(0, max_k(0, j));
        }
        //return max_score,max_k
    }


    nc::NdArray<float> CenterNetPostProcess::transform_preds(nc::NdArray<float> coords, nc::NdArray<float> center, float scale,
                                                             int output_size[2]){
        nc::NdArray<float> target_coords = nc::zeros<float>(coords.shape());
        nc::NdArray<float> target_coords_temp;

        cv::Mat trans = get_affine_transform(center, scale, 0, output_size, {0, 0}, 1);
        nc::NdArray<float> trans_NdArray = nc::zeros<float>(trans.rows, trans.cols);
        double* ptr_data = (double*)trans.data;
        for (int i = 0; i < trans.rows; i++)
        {
            for (int j = 0; j < trans.cols; j++)
            {
                trans_NdArray(i,j) = (float)ptr_data[i*trans.cols+j];
            }
        }
        //trans_NdArray.astype<float>();
        for (int p = 0; p < coords.shape().rows; p++)
        {
            target_coords_temp = nc::copy(affine_transform(coords(p, {0,2}), trans_NdArray));
            for (int q = 0; q < 2; q++)
            {
                target_coords(p, q) = target_coords_temp(q,0);
            }
        }
        return target_coords;
    }

    std::vector<nc::NdArray<float>> CenterNetPostProcess::ctdet_post_process(nc::NdArray<float> dets,
                                            nc::NdArray<float> c, float s, int h, int w, int num_classes){
        int w_h[2]={w,h};
        nc::NdArray<float> dets_01 = transform_preds(dets(dets.rSlice(), {0, 2}), c, s, w_h);
        nc::NdArray<float> dets_23 = transform_preds(dets(dets.rSlice(), {2, 4}), c, s, w_h);
        nc::NdArray<float> classes = dets(dets.rSlice(), {5, 6});
        nc::NdArray<float> scores = dets(dets.rSlice(), {4, 5});
        std::vector<nc::NdArray<float>> ret;
        nc::NdArray<float> dets_cat = nc::concatenate({dets_01, dets_23, dets(dets.rSlice(), {4, 5})}, nc::Axis::COL);
	


        for (int i = 0; i < cat; i++)
        {
            int sum_same = 0;
            nc::NdArray<float> inds = nc::zeros<float>(K, 1);//标置位
            for (int j = 0; j < K; j++) {
                if (int(classes(j, 0)) == i)//相等则置1，否则为0
                {
                    inds(j,0) = 1;
                    sum_same++;
                }
            }

            nc::NdArray<float> dets_post = nc::zeros<float>(K, 5);//标置位
            for (int j = 0; j < K; j++)
            {
                for (int k = 0; k < 5; k++)//(boxes=4)+1+1
                {
                    dets_post(j, k) = dets_cat(j, k) * inds(j, 0);
                }
            }
            //nc::append(ret,dets_post,nc::Axis::COL);
	    //

            ret.push_back(dets_post);
	    
        }

        return ret; // [80, 100, 5]
    }


    std::vector<nc::NdArray<float>> CenterNetPostProcess::post_process(nc::NdArray<float> dets, Meta meta) {
    //dets = dets.reshape(-1, dets.shape[1]);
        nc::NdArray<float> c = meta.c;
        float s = meta.s;
        int out_height = meta.out_height;
        int out_width = meta.out_width;
        std::vector<nc::NdArray<float>> dets_result;

        dets_result = ctdet_post_process(dets, c, s, out_height, out_width, cat);
        return dets_result;
    }


    nc::NdArray<float> CenterNetPostProcess::ctdet_decode(nc::NdArray<float> heat, nc::NdArray<float> wh,
                                                          nc::NdArray<float> reg, bool cat_spec_wh, int K)
    {
    
        //cat, size = heat.shape
        //scores, inds, clses, ys, xs = _topk(heat, K=K)
        nc::NdArray<float> scores = nc::zeros<float>(1, K);
        nc::NdArray<uint32_t> inds = nc::zeros<uint32_t>(1, K);
        nc::NdArray<uint32_t> clses = nc::zeros<uint32_t>(1, K);
        nc::NdArray<uint32_t> ys = nc::zeros<uint32_t>(cat, K);
        nc::NdArray<uint32_t> xs = nc::zeros<uint32_t>(cat, K);
        _topk(heat, K, scores, inds, clses, ys, xs);
    
        nc::NdArray<float> XS;
        nc::NdArray<float> YS;
        if (reg.size())//非空
        {
            _tranpose_and_gather_feat(reg, inds);
            reg.reshape(K, 2);
            nc::NdArray<float> xs_float = xs.reshape(K, 1).astype<float>();
            XS = xs_float + reg(reg.rSlice(),{0, 1});
            nc::NdArray<float> ys_float = ys.reshape(K, 1).astype<float>();
            YS = ys_float + reg(reg.rSlice(),{1, 2});
        }
        else
        {
            nc::NdArray<float> xs_float = xs.reshape(K, 1).astype<float>();
            XS = xs_float + float(0.5);
            nc::NdArray<float> ys_float = ys.reshape(K, 1).astype<float>();
            YS = ys_float + float(0.5);
        }
    
        _tranpose_and_gather_feat(wh, inds);
        if (!cat_spec_wh) {
            wh.reshape(K, 2);
        }
        nc::NdArray<float> clses_float  = clses.reshape(K, 1).astype<float>();
        scores.reshape(K, 1);
        nc::NdArray<float> bboxes = nc::concatenate({XS - wh(wh.rSlice(), {0, 1}) / float(2.0),
                                  YS - wh(wh.rSlice(),{1, 2}) / float(2.0),
                                  XS + wh(wh.rSlice(),{0, 1}) / float(2.0),
                                  YS + wh(wh.rSlice(),{1, 2}) / float(2.0)}, nc::Axis::COL);
        nc::NdArray<float> detections = nc::concatenate({bboxes, scores, clses_float}, nc::Axis::COL);
	return detections; // [100, 5]
    
    }


    void CenterNetPostProcess::GenerateBoxes(std::vector<nc::NdArray<float>> results,
                                             std::vector <MxBase::ObjectInfo> &detBoxes) {
        float maxr = 0;
        for (int i = 0; i < results.size(); i++) { // size = 80
            for (int j = 0; j < results[i].shape().rows; j++) {
                if (maxr < results[i](j,4)) maxr = results[i](j,4);
                if (results[i](j,4) > scoreThresh_) { //置信度阈值
                    MxBase::ObjectInfo det;
                    det.x0 = results[i](j, 0);
                    det.x1 = results[i](j, 2);
                    det.y0 = results[i](j, 1)+0.5;
                    det.y1 = results[i](j, 3)+0.5;
                    det.confidence = results[i](j,4);
                    det.classId = i;
		    det.className = configData_.GetClassName(i);
                    detBoxes.emplace_back(det);
                }
            }
        }
        if (maxr < scoreThresh_) {
            MxBase::ObjectInfo det;
            det.x0 = 0;
            det.x1 = 0;
            det.y0 = 0;
            det.y1 = 0;
            det.confidence = maxr;
            det.classId = cat + 1;
            detBoxes.emplace_back(det);
        }
    }


    void CenterNetPostProcess::ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
                                                     std::vector<std::vector<ObjectInfo>> &objectInfos,
                                                     const std::vector<ResizedImageInfo> &resizedImageInfos) {
        LogInfo << "CenterNetPostProcess start to write results.";
        ResizedImageInfo resizedInfo = resizedImageInfos[0];
        if (tensors.size() == 0) {
            return;
        }
        auto shape = tensors[0].GetShape();
        if (shape.size() == 0) {
            return;
        }
        uint32_t batchSize = shape[0];
        for (uint32_t i = 0; i < batchSize; i++) {
            Meta meta;
            int widthResize = resizedImageInfos[i].widthResize;
            int heightResize = resizedImageInfos[i].heightResize;
            int widthOriginal = resizedImageInfos[i].widthOriginal;
            int heightOriginal = resizedImageInfos[i].heightOriginal;
            int widthResizeBeforePadding, heightResizeBeforePadding;
            if (widthOriginal > heightOriginal) {
                widthResizeBeforePadding = widthResize;
                heightResizeBeforePadding = static_cast<int>(static_cast<float>(widthResize) /
                        widthOriginal * heightOriginal);
            }
            else {
                heightResizeBeforePadding = heightResize;
                widthResizeBeforePadding = static_cast<int>(static_cast<float>(heightResize) /
                        heightOriginal * widthOriginal);
            }
            widthResizeBeforePadding = widthResize;
            heightResizeBeforePadding = heightResize;
	    //nc::NdArray<float> c = {float(widthResizeBeforePadding/2), float(heightResizeBeforePadding/2)};
            //float s = MAX(heightResizeBeforePadding, widthResizeBeforePadding) * 1.0;
	   // nc::NdArray<float> c = {176.0,115.0};
	    //float s = 352.0;
	    nc::NdArray<float> c = {float(widthOriginal/2), float(heightOriginal/2)};
            float s = MAX(heightOriginal, widthOriginal) * 1.0;
	    //float s = 256.0;
	    //nc::NdArray<float> c = {256,256};
//            s = s + (s / 512.0) * cat * 2;
            meta.c = c;
            meta.s = s;
            meta.out_width = 128;
            meta.out_height = 128;

            int size_feature = 128 * 128;
            nc::NdArray<float> hm = nc::zeros<float>(cat, size_feature);//heatmap
//            nc::NdArray<float> heat = nc::zeros<float>(cat, size_feature);//处理过的heatmap
            nc::NdArray<float> wh = nc::zeros<float>(2, size_feature);//width,height
            nc::NdArray<float> reg = nc::zeros<float>(2, size_feature);//回归的位置，偏移?
            LogInfo << "Until here is good.";
            ReadDataFromTensor(tensors, hm, wh, reg);
            //ReadDataFromBin(P1, P2, P3, hm, wh, reg);
            nc::NdArray<float> dets = ctdet_decode(hm, wh, reg, false, K);
            std::vector<nc::NdArray<float>> result_dets = post_process(dets, meta);
	    
            // generate bounding boxes
            std::vector<ObjectInfo> objectInfo;
            GenerateBoxes(result_dets, objectInfo);
            objectInfos.push_back(objectInfo);
        }
        LogInfo << "CenterNetPostProcess write results successed.";
    }

    APP_ERROR CenterNetPostProcess::Process(const std::vector<TensorBase> &tensors,
                                               std::vector<std::vector<ObjectInfo>> &objectInfos,
                                               const std::vector<ResizedImageInfo> &resizedImageInfos,
                                               const std::map<std::string, std::shared_ptr<void>> &paramMap) {
        LogInfo << "Start to Process CenterNetPostProcess.";
        APP_ERROR ret = APP_ERR_OK;
        if (resizedImageInfos.size() == 0) {
            ret = APP_ERR_INPUT_NOT_MATCH;
            LogError << GetError(ret) << "resizedImageInfos is not provided which is necessary "
                                         "for CenterNetPostProcess.";
            return ret;
        }
        auto inputs = tensors;
        ret = CheckAndMoveTensors(inputs);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "CheckAndMoveTensors failed.";
            return ret;
        }
        ObjectDetectionOutput(inputs, objectInfos, resizedImageInfos);
        LogObjectInfos(objectInfos);
        LogInfo << "End to Process CenterNetPostProcess.";
        return APP_ERR_OK;
    }


    extern "C" {
    std::shared_ptr<MxBase::CenterNetPostProcess> GetObjectInstance() {
        LogInfo << "Begin to get CenterNetPostProcess instance.";
        auto instance = std::make_shared<MxBase::CenterNetPostProcess>();
        LogInfo << "End to get CenterNetPostProcess instance.";
        return instance;
    }
    }
}
