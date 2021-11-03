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

#include "MxBase/Log/Log.h"
#include "carplate_recognition.h"


/*@brief: 初始化结构体参数
* @param：initParam：模型参数结构体
* @retval:none
*/
void InitCarPlateRecognitionParam(InitParam &initParam)
{  
    initParam.checkTensor = true;
    initParam.deviceId = 0;
    initParam.DetecModelPath = "./model/retinaface.om";    
    initParam.RecogModelPath = "./model/lpr.om";
    initParam.nmsThreshold = 0.4;
    initParam.scoreThreshold = 0.6;
    initParam.width = 640;
    initParam.height= 640;
    initParam.steps= {8, 16, 32};
    initParam.min_sizes = {{24, 48}, {96, 192}, {384, 768}};
    initParam.variances = {0.1, 0.2};
    initParam.scale = {640, 640, 640, 640};
}


/*@brief: 主函数
* @param：argc：参数的个数（本程序的argc=2）
* @param：argv：参数向量，其中argv[0]是程序的路径，argv[1]是待检测图片的存放路径
*/
int main(int argc, char* argv[])
{
    if (argc <= 1) { 
        LogWarn << "Please input image path, such as './imgs/test.jpg'.";
        return APP_ERR_OK;
    }

    InitParam initParam;
    InitCarPlateRecognitionParam(initParam); 
	std::string imgPath = argv[1];
	
	// 创建一个车牌识别对象并初始化
	auto car_plate_recognition = std::make_shared<CarPlate_Recognition>();
	APP_ERROR ret = car_plate_recognition->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "CarPlate_Recognition init failed, ret=" << ret << ".";
        return ret;
    }

	// 进行车牌识别流程
    ret = car_plate_recognition->Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "CarPlate_Recognition process failed, ret=" << ret << ".";
        car_plate_recognition->DeInit();
        return ret;
    }

	// 识别完成，将资源释放
    car_plate_recognition->DeInit();

    return APP_ERR_OK;
}
