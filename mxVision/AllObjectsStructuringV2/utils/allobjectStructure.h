#ifndef MAIN_ALLOBJECTSTRUCTURE_H
#define MAIN_ALLOBJECTSTRUCTURE_H
#include "mxBase/MxBase.h"
#include "mxBase/postprocess/include/ObjectPostProcessors/Yolov3PostProcess.h"
#include "../postprocess/faceLandmark/FaceLandmarkPostProcess.h"
#include "../posrprocess/faceAlignment/FaceAlignment.h"

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
}

struct FrameImage
{
    MxBase::Image image;
    uint32_t frameID = 0;
    uint32_t channelID = 0;
}

#endif // MAIN_ALLOBJECTSTRUCTURE_H