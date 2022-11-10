#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 
set -e 
current_folder="$( cd "$(dirname "$0")" ;pwd -P )"


SAMPLE_FOLDER=(
    # ActionRecognition/
    # CrowdCounting/
    # mxBase_wheatDetection/
    # EdgeDetectionPicture/
    HelmetIdentification/
    Individual/
    # human_segmentation/
    # OpenposeKeypointDetection/
    PersonCount/
    FatigueDrivingRecognition/
    # CartoonGANPicture/
    # HeadPoseEstimation/
    FaceBoxes/
    BertTextClassification/
    # RTM3DTargetDetection/
    EfficientDet/
    SentimentAnalysis/
    # RotateObjectDetection/
    FairMOT/
    UltraFastLaneDetection/
    VehicleIdentification/
    yunet/
   RoadSegmentation/
    PassengerflowEstimation/
    VehicleRetrogradeRecognition/
    Collision/
    PassengerflowEstimation/
    CenterFace/
    YOLOX/
    PicoDet/
  SOLOV2/
  OpenCVPlugin/
    RefineDet/
    FCOS/
)

err_flag=0
for sample in ${SAMPLE_FOLDER[@]};do
    cd ${current_folder}/${sample}
    bash build.sh || {
        echo -e "Failed to build ${sample}"
        err_flag=1
    }
done


if [ ${err_flag} -eq 1 ]; then
    exit 1
fi
exit 0
<<<<<<< HEAD
=======

	PicoDet/
  SOLOV2/
  OpenCVPlugin/
  X3D/
)

err_flag=0
for sample in ${SAMPLE_FOLDER[@]};do
    cd ${current_folder}/${sample}
    bash build.sh || {
        echo -e "Failed to build ${sample}"
		err_flag=1
    }
done


if [ ${err_flag} -eq 1 ]; then
	exit 1
fi
exit 0
>>>>>>> 515cf51749a227c1871c514c2ee53573a59247a6
