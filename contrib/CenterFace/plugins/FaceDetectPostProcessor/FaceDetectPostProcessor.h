/*
 * Copyright(C) 2021 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CENTERFACE_POSTPROCESSOR_H_
#define CENTERFACE_POSTPROCESSOR_H_
#include <vector>
#include <string>
#include <map>
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/ModelPostProcessors/ModelPostProcessorBase/ObjectPostDataType.h"
#include "MxBase/Log/Log.h"
struct FaceInfo {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  float landmarks[10];
};
namespace MxBase {
class FaceDetectPostProcessor : public MxBase::ObjectPostProcessBase {
public:
  APP_ERROR
  Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) override;

  APP_ERROR DeInit() override;

  APP_ERROR
  Process(const std::vector<MxBase::TensorBase> &tensors,
          std::vector<std::vector<MxBase::ObjectInfo>> &objectInfos,
          const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos = {},
          const std::map<std::string, std::shared_ptr<void>> &configParamMap =
              {}) override;

  APP_ERROR detect(std::vector<void *> &featLayerData, std::vector<FaceInfo> &faces,
                   const ImageInfo &imgInfo);

private:
  enum NmsMethod { LINEAR = 1, GAUSSIAN = 2, ORIGINAL_NMX = 3 };
  int modelWidth_ = 200;
  int modelHeight_ = 200;
  float scale_w = 1.f;
  float scale_h = 1.f;
  int DOWN_SAMPLE = 4;
  int SCALE_FACTOR = 2;
  // IOU thresh hold
  float nmsThresh_ = 0.4;
  int nmsMethod = 1;

private:
  void nms(std::vector<FaceInfo> &vec_boxs, unsigned int method = 1, float sigma = 0.5);
  APP_ERROR decode(float *heatmap, float *scale, float *offset, float *landmarks,
              std::vector<FaceInfo> &faces, const ImageInfo &imageinfo);
  std::vector<int> getIds(float *heatmap, int h, int w);
  float GetNmsWeight(float iou, float sigma, int method);
  float GetIou(FaceInfo &curr_box, FaceInfo *max_ptr, float overlaps);
  void squareBox(std::vector<FaceInfo> &faces, const ImageInfo &imageinfo);
  APP_ERROR ProcessOnePicture(std::vector<void *> &featLayerData,
                              std::vector<MxBase::ObjectInfo> &objInfos,
                              const MxBase::ResizedImageInfo &resizeInfo);
};

extern "C" {
std::shared_ptr<MxBase::FaceDetectPostProcessor> GetObjectInstance();
}
} // namespace MxBase

#endif // CENTERFACE_POSTPROCESSOR_H_
