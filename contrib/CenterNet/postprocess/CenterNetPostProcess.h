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

#ifndef CENTERNET_POST_PROCESS_H
#define CENTERNET_POST_PROCESS_H
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "opencv2/opencv.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Functions.hpp"

namespace DefaultValues {
    const int DEFAULT_CLASS_NUM = 80;
    const float DEFAULT_SCORE_THRESH = 0.0;
}

namespace MxBase {
    class CenterNetPostProcess: public ObjectPostProcessBase {

    public:
        CenterNetPostProcess() = default;

        ~CenterNetPostProcess() = default;

        CenterNetPostProcess(const CenterNetPostProcess &other);

        CenterNetPostProcess &operator=(const CenterNetPostProcess &other);

        APP_ERROR Init(const std::map <std::string, std::shared_ptr<void>> &postConfig);

        APP_ERROR DeInit();

        APP_ERROR Process(const std::vector <MxBase::TensorBase> &tensors,
                          std::vector <std::vector<MxBase::ObjectInfo>> &objectInfos,
                          const std::vector <MxBase::ResizedImageInfo> &resizedImageInfos = {},
                          const std::map <std::string, std::shared_ptr<void>> &paramMap = {});
        struct Meta{
            nc::NdArray<float> c;
            float s;
            int out_height;
            int out_width;
        };

    protected:
        bool IsValidTensors(const std::vector <MxBase::TensorBase> &tensors) const;

        /**
         * @brief Obtain the detection bboxes by post-processing the inference result of the object detection model
         * @param tensors - Regression tensor and classification tensor output from the model inference plugin
         * @param resizedImageInfos - Image information obtained from the mxpi_imageresize plugin (including the
         * width and height of the original image and those of the zoomed image
         * @param objectInfos - Vector of vector of MxBase::ObjectInfo, which stores the information of detected
         * bounding boxes of each input image
         * */
        void ObjectDetectionOutput(const std::vector <MxBase::TensorBase> &tensors,
                                   std::vector <std::vector<MxBase::ObjectInfo>> &objectInfos,
                                   const std::vector <MxBase::ResizedImageInfo> &resizedImageInfos = {});

        void ReadDataFromTensor(const std::vector <MxBase::TensorBase> &tensors,
                                nc::NdArray<float> &heatmap,
                                nc::NdArray<float> &wh,
                                nc::NdArray<float> &regression);
        
        nc::NdArray<float> _gather_feat(nc::NdArray<float> feat, nc::NdArray<uint32_t> ind);
        nc::NdArray<uint32_t> _gather_feat(nc::NdArray<uint32_t> feat, nc::NdArray<uint32_t> ind);
        void _tranpose_and_gather_feat(nc::NdArray<float> &feat, nc::NdArray<uint32_t> ind);
        void _topk(nc::NdArray<float> heat, int K, nc::NdArray<float> &topk_score,
                   nc::NdArray<uint32_t> &topk_inds, nc::NdArray<uint32_t> &topk_clses,
                   nc::NdArray<uint32_t> &topk_ys, nc::NdArray<uint32_t> &topk_xs);
                   
        nc::NdArray<float> get_3rd_point(nc::NdArray<float> a, nc::NdArray<float> b);
        nc::NdArray<float> get_dir(nc::NdArray<float> src_point, float rot_rad);
        cv::Mat get_affine_transform(nc::NdArray<float> center, float scale, float rot, int output_size[2],
                                     nc::NdArray<float> shift, int inv);
        nc::NdArray<float> affine_transform(nc::NdArray<float> pt, nc::NdArray<float> t);
        void naive_arg_topK_3d(nc::NdArray<float> matrix, int K, nc::NdArray<float> &max_score,
                               nc::NdArray<uint32_t> &max_k);
        void naive_arg_topK_2d(nc::NdArray<float> matrix, int K, nc::NdArray<float> &max_score, nc::NdArray<uint32_t> &max_k);
        nc::NdArray<float> transform_preds(nc::NdArray<float> coords, nc::NdArray<float> center, float scale,
                                           int output_size[2]);
        std::vector<nc::NdArray<float>> ctdet_post_process(nc::NdArray<float> dets,
                                                           nc::NdArray<float> c, float s, int h, int w, int num_classes);
        std::vector<nc::NdArray<float>> post_process(nc::NdArray<float> dets, Meta meta);
        nc::NdArray<float> ctdet_decode(nc::NdArray<float> heat, nc::NdArray<float> wh,
                                         nc::NdArray<float> reg, bool cat_spec_wh, int K);
        void GenerateBoxes(std::vector<nc::NdArray<float>> result,
                           std::vector <MxBase::ObjectInfo> &detBoxes);

    protected:
        int classNum_ = DefaultValues::DEFAULT_CLASS_NUM;
        float scoreThresh_ = DefaultValues::DEFAULT_SCORE_THRESH; // Confidence threhold

    };
    extern "C" {
    std::shared_ptr<MxBase::CenterNetPostProcess> GetObjectInstance();
    }
}
#endif
