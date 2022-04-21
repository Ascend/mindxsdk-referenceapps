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
#include <queue>
#include "MxpiTrackRetrogradeCar.h"
#include "MxBase/Log/Log.h"
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
namespace {
    const string SAMPLE_KEY = "MxpiObjectList";
    const string SAMPLE_KEY2 = "MxpiTrackLetList";
    static std::vector<std::queue<center>> pts(10000);  // 保存每个车辆轨迹的最新的20个bbox的中心点
    const float count_center =2.0;
    static std::vector<int> is_retrograde;
}
bool is_element_in_vector(vector<int> v,int element){
    vector<int>::iterator it;
    it=find(v.begin(),v.end(),element);
    if (it!=v.end()){
        return true;
    }
    else{
        return false;
    }
}

APP_ERROR MxpiTrackRetrogradeCar::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "MxpiTrackRetrogradeCar::Init start.";
    APP_ERROR ret = APP_ERR_OK;

    // Get the property values by key
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get();
    std::shared_ptr<string> motNamePropSptr = std::static_pointer_cast<string>(configParamMap["motSource"]);
    motName_ = *motNamePropSptr.get();   
    std::shared_ptr<int> inputX1PropSptr = std::static_pointer_cast<int>(configParamMap["x1"]);
    limit_x1 = *inputX1PropSptr.get(); 
    std::shared_ptr<int> inputY1PropSptr = std::static_pointer_cast<int>(configParamMap["y1"]);
    limit_y1 = *inputY1PropSptr.get(); 
    std::shared_ptr<int> inputX2PropSptr = std::static_pointer_cast<int>(configParamMap["x2"]);
    limit_x2 = *inputX2PropSptr.get(); 
    std::shared_ptr<int> inputY2PropSptr = std::static_pointer_cast<int>(configParamMap["y2"]);
    limit_y2 = *inputY2PropSptr.get(); 
    std::shared_ptr<int> isVerticalPropSptr = std::static_pointer_cast<int>(configParamMap["isVertical"]);
    is_vertical = *isVerticalPropSptr.get(); 
    std::shared_ptr<string> descriptionMessageProSptr = 
        std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
    descriptionMessage_ = *descriptionMessageProSptr.get();
    return APP_ERR_OK;
}

APP_ERROR MxpiTrackRetrogradeCar::DeInit()
{
    LogInfo << "MxpiTrackRetrogradeCar::DeInit end.";
    return APP_ERR_OK;
}

APP_ERROR MxpiTrackRetrogradeCar::SetMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,
    const MxpiErrorInfo mxpiErrorInfo)
{
    APP_ERROR ret = APP_ERR_OK;
    // Define an object of MxpiMetadataManager
    MxpiMetadataManager mxpiMetadataManager(buffer);
    ret = mxpiMetadataManager.AddErrorInfo(pluginName, mxpiErrorInfo);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to AddErrorInfo.";
        return ret;
    }
    ret = SendData(0, buffer);
    return ret;
}

APP_ERROR MxpiTrackRetrogradeCar::PrintMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,
    MxpiErrorInfo mxpiErrorInfo, APP_ERROR app_error, std::string errorName)
{
    ErrorInfo_ << GetError(app_error, pluginName_) << errorName;
    LogError << errorName;
    mxpiErrorInfo.ret = app_error;
    mxpiErrorInfo.errorInfo = ErrorInfo_.str();
    SetMxpiErrorInfo(buffer, pluginName_, mxpiErrorInfo);
    return app_error;
}



/*
 * @description: Replace className with trackId 
 */
APP_ERROR MxpiTrackRetrogradeCar::GenerateSampleOutput(const MxpiObjectList srcMxpiObjectList, 
                                                            const MxpiTrackLetList srcMxpiTrackLetList, 
                                                            MxpiObjectList& dstMxpiObjectList)
{
    double k;// 分界线斜率
    double b;
    k = double(limit_y1-limit_y2)/double(limit_x1-limit_x2);
    b = double(limit_y1)-k*double(limit_x1);
    
    for (int i = 0; i < srcMxpiObjectList.objectvec_size(); i++){
        MxpiObject srcMxpiObject = srcMxpiObjectList.objectvec(i);       
        MxpiClass srcMxpiClass = srcMxpiObject.classvec(0);

        // 获取bounding box的中心位置
        center boxs = {(srcMxpiObject.x0()+srcMxpiObject.x1())/count_center, (srcMxpiObject.y0()+srcMxpiObject.y1())/count_center};
        for(int j = 0; j < srcMxpiTrackLetList.trackletvec_size(); j++){
            MxpiTrackLet srcMxpiTrackLet = srcMxpiTrackLetList.trackletvec(j);  
            int index = (int)srcMxpiTrackLet.trackid();
            // 保存每个车辆轨迹最新的20个bbox
            if(srcMxpiTrackLet.trackflag() != 2){
                MxpiMetaHeader srcMxpiHeader = srcMxpiTrackLet.headervec(0);  
                if(srcMxpiHeader.memberid() == i){
                    if(pts[index].size()>=20){
                        pts[index].pop();
                        pts[index].push(boxs);
                    }
                    else{
                        pts[index].push(boxs);
                    }
                    std::vector<center> last_point = {};
                    for(uint32_t j = 0; j < pts[index].size(); j++){
                        if(pts[index].size()-j<=2){
                            last_point.push_back(pts[index].front());
                        }
                        pts[index].push(pts[index].front());
                        pts[index].pop();
                    }
                    // 不少于2个bbox的车辆轨迹可用于方向判断
                    if(last_point.size()==2){
                        int p1,p2,m1,m2;
                        // 去掉边缘误差判断情况
                        if(last_point[0].x<30 || last_point[0].x>1250 || last_point[1].x<30 || last_point[1].x>1250 || last_point[0].y<30 || last_point[0].y>690 || last_point[1].y<30 || last_point[1].y>690)
                            continue;
                        // 如果计数标志位是垂直的，使用x坐标来判断
                        if(is_vertical){
                            p1=last_point[0].x;
                            p2=last_point[1].x;
                            m1=last_point[0].y;
                            m2=last_point[1].y;
                            if(p1>p2){
                                if((limit_y1+limit_y2)/2 < m1){ // 逆行
                                    MxpiObject* dstMxpiObject = dstMxpiObjectList.add_objectvec();    
                                    dstMxpiObject->set_x0(srcMxpiObject.x0());
                                    dstMxpiObject->set_y0(srcMxpiObject.y0());
                                    dstMxpiObject->set_x1(srcMxpiObject.x1());
                                    dstMxpiObject->set_y1(srcMxpiObject.y1());
                                    MxpiClass* dstMxpiClass = dstMxpiObject->add_classvec();   
                                    dstMxpiClass->set_confidence(srcMxpiClass.confidence()); 
                                    dstMxpiClass->set_classid(0);
                                    dstMxpiClass->set_classname(to_string(srcMxpiTrackLet.trackid()));
                                }
                            }
                            else{
                                if((limit_y1+limit_y2)/2 > m1){ // 逆行
                                    MxpiObject* dstMxpiObject = dstMxpiObjectList.add_objectvec();    
                                    dstMxpiObject->set_x0(srcMxpiObject.x0());
                                    dstMxpiObject->set_y0(srcMxpiObject.y0());
                                    dstMxpiObject->set_x1(srcMxpiObject.x1());
                                    dstMxpiObject->set_y1(srcMxpiObject.y1());
                                    MxpiClass* dstMxpiClass = dstMxpiObject->add_classvec();   
                                    dstMxpiClass->set_confidence(srcMxpiClass.confidence()); 
                                    dstMxpiClass->set_classid(0);
                                    dstMxpiClass->set_classname(to_string(srcMxpiTrackLet.trackid()));
                                }
                            }
                        }
                        else{
                            p1=last_point[0].y;
                            p2=last_point[1].y;
                            m1=last_point[0].x;
                            m2=last_point[1].x;
                            if(p1>p2){ // 向上开
                                if((k*m1+b > p1 && k<0) || (k*m1+b < p1 && k>0)){ // 逆行
                                    if(is_element_in_vector(is_retrograde, srcMxpiTrackLet.trackid())){
                                        MxpiObject* dstMxpiObject = dstMxpiObjectList.add_objectvec();    
                                        dstMxpiObject->set_x0(srcMxpiObject.x0());
                                        dstMxpiObject->set_y0(srcMxpiObject.y0());
                                        dstMxpiObject->set_x1(srcMxpiObject.x1());
                                        dstMxpiObject->set_y1(srcMxpiObject.y1());
                                        MxpiClass* dstMxpiClass = dstMxpiObject->add_classvec();   
                                        dstMxpiClass->set_confidence(srcMxpiClass.confidence()); 
                                        dstMxpiClass->set_classid(0);
                                        dstMxpiClass->set_classname(to_string(srcMxpiTrackLet.trackid()));
                                    }
                                    else
                                        is_retrograde.push_back(srcMxpiTrackLet.trackid());
                                }
                            }
                            if(p1<p2){
                                if((k*m1+b < p1 && k<0) || (k*m1+b > p1 && k>0)){ // 逆行
                                    if(is_element_in_vector(is_retrograde, srcMxpiTrackLet.trackid())){
                                        MxpiObject* dstMxpiObject = dstMxpiObjectList.add_objectvec();    
                                        dstMxpiObject->set_x0(srcMxpiObject.x0());
                                        dstMxpiObject->set_y0(srcMxpiObject.y0());
                                        dstMxpiObject->set_x1(srcMxpiObject.x1());
                                        dstMxpiObject->set_y1(srcMxpiObject.y1());
                                        MxpiClass* dstMxpiClass = dstMxpiObject->add_classvec();   
                                        dstMxpiClass->set_confidence(srcMxpiClass.confidence()); 
                                        dstMxpiClass->set_classid(0);
                                        dstMxpiClass->set_classname(to_string(srcMxpiTrackLet.trackid()));
                                    }
                                    else
                                        is_retrograde.push_back(srcMxpiTrackLet.trackid());
                                }
                            }
                        }

                    }
                    continue;
                }
            }
        }
    }
    return APP_ERR_OK;
}

APP_ERROR MxpiTrackRetrogradeCar::Process(std::vector<MxpiBuffer*>& mxpiBuffer){
    LogInfo << "MxpiTrackRetrogradeCar::Process start";
    MxpiBuffer* buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, APP_ERR_COMM_FAILURE, "MxpiTrackRetrogradeCar process is not implemented");
    }
    shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(parentName_);  // Get the data from buffer
    shared_ptr<void> metadata2 = mxpiMetadataManager.GetMetadata(motName_);
    if (metadata == nullptr) {
        shared_ptr<MxpiObjectList> dstMxpiObjectListSptr = make_shared<MxpiObjectList>(); 
        MxpiObject* dstMxpiObject = dstMxpiObjectListSptr->add_objectvec();   
        MxpiClass* dstMxpiClass = dstMxpiObject->add_classvec();    
        APP_ERROR ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiObjectListSptr));
        if (ret != APP_ERR_OK) {
            return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, ret, "MxpiTrackRetrogradeCar add metadata failed.");
        }
        SendData(0, *buffer); // Send the data to downstream plugin
        LogInfo << "MxpiTrackRetrogradeCar::Process end";
        return APP_ERR_OK;
    }
    google::protobuf::Message* msg = (google::protobuf::Message*)metadata.get();
    const google::protobuf::Descriptor* desc = msg->GetDescriptor();
    google::protobuf::Message* msg2 = (google::protobuf::Message*)metadata2.get();
    const google::protobuf::Descriptor* desc2 = msg2->GetDescriptor();
    if (desc->name() != SAMPLE_KEY) {    // check whether the proto struct name is MxpiObjectList
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, APP_ERR_PROTOBUF_NAME_MISMATCH, "Proto struct name is not MxpiObjectList, failed");
    }
    if (desc2->name() != SAMPLE_KEY2) {   // check whether the proto struct name is MxpiTrackList
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, APP_ERR_PROTOBUF_NAME_MISMATCH, "Proto struct name is not MxpiTrackLetList, failed");
    }
    shared_ptr<MxpiObjectList> srcMxpiObjectListSptr = static_pointer_cast<MxpiObjectList>(metadata);
    shared_ptr<MxpiTrackLetList> srcMxpiTrackLetListSptr = static_pointer_cast<MxpiTrackLetList>(metadata2);
    shared_ptr<MxpiObjectList> dstMxpiObjectListSptr = make_shared<MxpiObjectList>();    
    APP_ERROR ret = GenerateSampleOutput(*srcMxpiObjectListSptr,*srcMxpiTrackLetListSptr,*dstMxpiObjectListSptr); // Generate sample output
    if (ret != APP_ERR_OK) {
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, ret, "MxpiTrackRetrogradeCar gets inference information failed.");
    }
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiObjectListSptr)); // Add Generated data to metedata
    if (ret != APP_ERR_OK) {
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, ret, "MxpiTrackRetrogradeCar add metadata failed.");
    }
    SendData(0, *buffer);  // Send the data to downstream plugin
    LogInfo << "MxpiTrackRetrogradeCar::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiTrackRetrogradeCar::DefineProperties()
{
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
   
    auto parentNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
        STRING, "dataSource", "inputName", "the name of postprocessor", "mxpi_distributor0_0", "NULL", "NULL"});
    auto motNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
        STRING, "motSource", "parentName", "the name of previous plugin", "mxpi_motsimplesortV20", "NULL", "NULL"});
    auto limitX1ProSptr = (std::make_shared<ElementProperty<int>>)(ElementProperty<int>{
            INT, "x1", "inputX1Value", "the point of x1", 1, -1, 1281});
    auto limitY1ProSptr = (std::make_shared<ElementProperty<int>>)(ElementProperty<int>{
            INT, "y1", "inputY1Value", "the point of y1", 0, -1, 721});
    auto limitX2ProSptr = (std::make_shared<ElementProperty<int>>)(ElementProperty<int>{
            INT, "x2", "inputX2Value", "the point of x2", 0, -1, 1281});
    auto limitY2ProSptr = (std::make_shared<ElementProperty<int>>)(ElementProperty<int>{
            INT, "y2", "inputY2Value", "the point of y2", 0, -1, 721});
    auto isVerticalProSptr = (std::make_shared<ElementProperty<int>>)(ElementProperty<int>{
            INT, "isVertical", "is the vedio vertical", "is the vedio vertical", 0, -1, 3});
    auto descriptionMessageProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
        STRING, "descriptionMessage", "message", "Description mesasge of plugin",  "This is MxpiTrackRetrogradeCar", "NULL", "NULL"});

    properties.push_back(parentNameProSptr);
    properties.push_back(motNameProSptr);
    properties.push_back(limitX1ProSptr);
    properties.push_back(limitX2ProSptr);
    properties.push_back(limitY1ProSptr);
    properties.push_back(limitY2ProSptr);
    properties.push_back(isVerticalProSptr);
    properties.push_back(descriptionMessageProSptr);
    return properties;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiTrackRetrogradeCar)

