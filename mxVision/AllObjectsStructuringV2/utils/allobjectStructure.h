#ifndef MAIN_ALLOBJECTSTRUCTURE_H
#define MAIN_ALLOBJECTSTRUCTURE_H
#include "MxBase/MxBase.h"
#include "MxBase/postprocess/include/ObjectPostProcessors/Yolov3PostProcess.h"
#include "../postprocessor/faceLandmark/FaceLandmarkPostProcess.h"
#include "../postprocessor/faceAlignment/FaceAlignment.h"

const int FPSSleep30 = 30;
const int YOLO_INPUT_WIDTH = 416;
const int YOLO_INPUT_HEIGHT = 416;
const int VEHICLE_ATTR_INPUT_WIDTH = 224;
const int VEHICLE_ATTR_INPUT_HEIGHT = 224;
const int CAR_PLATE_DETECT_INPUT_WIDTH = 480;
const int CAR_PLATE_DETECT_INPUT_HEIGHT = 640;
const int PED_ATTR_INPUT_WIDTH = 192;
const int PED_ATTR_INPUT_HEIGHT = 256;
const int PED_FEATURE_INPUT_WIDTH = 128;
const int PED_FEATURE_INPUT_HEIGHT = 384;
const int FACE_LANDMARK_INPUT_WIDTH = 96;
const int FACE_LANDMARK_INPUT_HEIGHT = 96;
const int CAR_PLATE_REC_INPUT_WIDTH = 272;
const int CAR_PLATE_REC_INPUT_HEIGHT = 72;
const int FACE_ALIGNMENT_INPUT_WIDTH = 112;
const int FACE_ALIGNMENT_INPUT_HEIGHT = 112;

const int EXECUTOR_NUM = 256;
const int INIT_RESOURCE_TIME = 5;
const int FRAME_WIDTH = 1920;
const int FRAME_HEIGHT = 1080;
const int SKIP_INTERVAL = 3;

enum SCENARIOS
{
    VEHICLE_ATTR,
    CAR_PLATE_DETECT,
    CAR_PLATE_RECOGNITION,
    PED_ATTR,
    PED_FEATURE,
    FACE_LANDMARK,
    FACE_ATTR,
    FACE_FEATURE
};

struct PreprocessedImage
{
    MxBase::Image image;
    SCENARIOS scenario;
    uint32_t frameID = 0;
    uint32_t channelID = 0;
    KeyPointAndAngle faceInfo;
};

struct FrameImage
{
    MxBase::Image image;
    uint32_t frameID = 0;
    uint32_t channelID = 0;
};

#endif // MAIN_ALLOBJECTSTRUCTURE_H