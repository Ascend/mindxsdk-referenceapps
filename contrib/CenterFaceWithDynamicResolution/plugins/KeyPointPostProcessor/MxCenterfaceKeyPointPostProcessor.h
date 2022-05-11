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

#ifndef OFFICIAL_CV_CENTERFACE_INFER_MXBASE_MXCENTERFACEKEYPOINTPOSTPROCESSOR_H_
#define OFFICIAL_CV_CENTERFACE_INFER_MXBASE_MXCENTERFACEKEYPOINTPOSTPROCESSOR_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "MxBase/Log/Log.h"
#include "MxBase/ModelPostProcessors/ModelPostProcessorBase/ObjectPostDataType.h"
#include "MxBase/PostProcessBases/KeypointPostProcessBase.h"
struct FaceInfo {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  float landmarks[10];
};

namespace MxBase {
class MxCenterfaceKeyPointPostProcessor
    : public MxBase::KeypointPostProcessBase {
public:
  // @modify:used directly instead of called within framework
  APP_ERROR Init(const std::string &configPath, const std::string &labelPath);

  APP_ERROR
  Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) override;

  /*
   * @description: Do nothing temporarily.
   * @return APP_ERROR error code.
   */
  APP_ERROR DeInit() override {
    // do nothing for this derived class
    return APP_ERR_OK;
  }

  APP_ERROR
  Process(const std::vector<MxBase::TensorBase> &tensors,
          std::vector<std::vector<KeyPointDetectionInfo>> &keyPointInfos,
          const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
          const std::map<std::string, std::shared_ptr<void>> &configParamMap)
      override;

  APP_ERROR Process(std::vector<void *> &featLayerData,
                    std::vector<KeyPointDetectionInfo> &keyPointInfos,
                    const MxBase::ResizedImageInfo &resizeInfo);
  void detect(std::vector<void *> &featLayerData, std::vector<FaceInfo> &faces,
              const ImageInfo &imgInfo, float scoreThresh = 0.5,
              float nmsThresh = 0.3);

private:
  void nms(std::vector<FaceInfo> &vec_boxs, float nmsthresh, float iouthresh,
           float sigma = 0.5, unsigned int method = 1);
  void decode(float *heatmap, float *scale, float *offset, float *landmarks,
              std::vector<FaceInfo> &faces, const ImageInfo &imageinfo,
              float scoreThresh, float nmsThresh);
  std::vector<int> getIds(float *heatmap, int h, int w, float thresh);
  void squareBox(std::vector<FaceInfo> &faces, const ImageInfo &imageinfo);
  APP_ERROR ReadConfigParams();

private:
  enum NumsMethod { LINEAR = 1, GAUSSIAN = 2, ORIGINAL_NMX = 3 };
  int modelWidth_ = 200;
  int modelHeight_ = 200;
  float scale_w = 1.f;
  float scale_h = 1.f;
  int DOWN_SAMPLE = 4;
  int SCALE_FACTOR = 2;
  // IOU thresh hold
  float iouThresh_ = 0.4;
  int numsMethod = 1;
};

extern "C" {
std::shared_ptr<MxBase::MxCenterfaceKeyPointPostProcessor> GetKeypointInstance();
}
} // namespace MxBase

#endif // OFFICIAL_CV_CENTERFACE_INFER_MXBASE_MXCENTERFACEKEYPOINTPOSTPROCESSOR_H_
