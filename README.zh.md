中文|[英文](README.md)
# MindX SDK Reference Apps

[MindX SDK](https://www.hiascend.com/software/mindx-sdk) 是华为推出的软件开发套件(SDK)，提供极简易用、高性能的API和工具，助力昇腾AI处理器赋能各应用场景。

mxSdkReferenceApps是基于MindX SDK开发的参考样例。


## 版本说明

**请在[SDK产品选择页面](https://www.hiascend.com/software/mindx-sdk)选择您使用的产品后，通过下拉框选择支持的SDK版本并查看配套关系。**

>目前SDK分为：制造质检 [mxManufacture](https://www.hiascend.com/software/mindx-sdk/mxManufacture/community)、视觉分析 [mxVision](https://www.hiascend.com/software/mindx-sdk/mxVision/community)、检索聚类 [mxIndex](https://www.hiascend.com/software/mindx-sdk/mxIndex/community)。  
以上分支版本号相同，为适配不同的方向的SDK组件。

- **当前分支样例版本适配说明如下：**    
    | SDK版本 | CANN版本 |
    |---|---|
    | 2.0.4 | [5.0.4](https://www.hiascend.com/software/cann/commercial) | 

## 目录结构与说明
| 目录 | 说明 |
|---|---|
| [build](./build) | 用户贡献样例构建目录 |
| [contrib](./contrib) | 用户贡献样例目录 |
| [docs](./docs) | 文档目录 |
| [mxVision](./mxVision) | 官方应用样例目录 | 
| [tools](./tools) | 开发测试相关工具 | 
| [tutorials](./tutorials) | 官方开发样例和文档参考工程目录 | 

## 安装
**按照如下步骤搭建环境：**      
     (1) 在[昇腾文档](https://www.hiascend.com/document?tag=commercial-developere)中选择**CANN 软件安装指南**，点击进入文档。  
     (2) 根据文档了解整体流程并根据文档进行硬件及CANN软件安装。  
     (3) 在SDK下载页面中选择**MindX SDK {版本} {产品} 用户指南**。  
     (4) mxManufacture和mxVision需根据不同开发(使用MindStudio开发/使用命令行方式开发)中对应**环境准备**章节完成安装。

## 开发样例  
**根据以下表单，选择需要运行的样例，并按照readme部署相关样例和参考工程**
| 样例名称 | 简介 |
|---|---|
| [DvppWrapper接口样例](./tutorials/DvppWrapperSample) | 对图片实现编码，解码，缩放，抠图，以及把样例图片编码为264视频文件 |
| [图像检测样例](./tutorials/ImageDetectionSample) | c++和python版本的图像检测样例，可刷输出检测结果 |
| [c++图片输入yolov3样例](./tutorials/mxBaseSample) | c++语言的yolov3图像检测样例及yolov3的后处理模块开发 |
| [c++视频输入yolov3样例](./tutorials/mxBaseVideoSample) | 针对视频输入的c++版本yolov3样例 |
| [绘图单元使用样例](./tutorials/OsdSample) | 使用osd对图像进行自定义绘图的样例 |
| [输入输出插件使用演示](./tutorials/PipelineInputOutputSample) | 对多种输入输出方式进行演示的样例 |
| [元数据输出输出样例](./tutorials/protocolSample) | 如何自行编解码元数据的演示样例 |
| [SDK插件开发](./tutorials/mindx_sdk_plugin) | [4-1](docs/quickStart/4-1插件开发调试指导.md)章节对应的样例代码 |
| [模型后处理插件开发](./tutorials/SamplePostProcess) | [4-2](docs/quickStart/4-2模型后处理库(内置类型)开发调试指导.md)章节对应的演示代码 |
| [自定义proto结构体](./tutorials/sampleCustomProto) | [4-3](docs/quickStart/4-3挂载自定义proto结构体.md)章节对应的演示代码 |
| [自定义后处理插件开发](./tutorials/samplePluginPostProc) | [4-4](docs/quickStart/4-4模型Tensor数据处理&自定义模型后处理.md)章节对应的演示代码 |
1. SDK已支持的模型直接使用**mxpi_{类型}postprocessor**插件配置对应后处理so和参数即可([SDK2.0.4模型支持列表](https://support.huawei.com/enterprise/zh/doc/EDOC1100234263/8c42df9f))
2. 未支持的模型根据任务类型，选择SDK已经支持的后处理基类(目标检测，分类任务，语义分割，文本生成)([SDK2.0.4后处理基类](https://support.huawei.com/enterprise/zh/doc/EDOC1100234263/51b8b606))去派生一个新的子类，参考**模型后处理插件开发**进行开发。
3. 如果当前后处理基类所采用的数据结构无法满足需求，用户需要自行开发**自定义后处理插件开发**，或将tensor数据通过**元数据输出输出样例**输出至外部来处理

## 运行  
**根据以下表单，选择需要运行的样例，并按照readme进行第三方依赖的安装及样例下载运行**      
| 样例名称 | 语言 | 适配SDK版本 | 简介 |
|---|---|---|---|
| [动作识别](./contrib/ActionRecognition) | python | >=2.0.4 | 单人独处、逗留超时、快速移动、剧烈运动、离床检测、攀高检测六种应用场景 |
| [AI风景画](./contrib/ai_paint) | python | >=2.0.4 | 从结构化描述生成对应风景照片 |
| [自动语音识别](./contrib/ASR&KWR) | python | >=2.0.4 |端到端的自动语音识别(AutoSpeechRecognition)+文本关键词识别(KeyWordRecognition) |
| [文本分类](./contrib/BertTextClassification) | c++ | >=2.0.4 | 新闻文本分类类别：体育、健康、军事、教育、汽车 |
| [车牌识别](./contrib/CarPlateRecognition) | c++ | >=2.0.4 | 对图像中的车牌进行检测，并对检测到每一个车牌进行识别 |
| [卡通图像生成](./contrib/CartoonGANPicture) | c++ | >=2.0.4 | 通用场景下的jpg图片卡通化 |
| [黑白图像上色](./contrib/Colorization) | python | >=2.0.4 | 输入黑白图像，自动对黑白图像进行上色，还原彩色图像 |
| [人群计数](./contrib/CrowdCounting) | c++ | >=2.0.4 | 人群计数目标检测，输出可视化结果 |
| [驾驶员状态识别](./contrib/DriverStatusRecognition) | python | >=2.0.4 | 识别视频中的驾驶员状态 |
| [边缘检测](./contrib/EdgeDetectionPicture) | c++ | >=2.0.4 | 图像边缘提取，输出可视化结果 |
| [EfficientDet](./contrib/EfficientDet) | python | >=2.0.4 | 使用 EfficientDet 模型进行目标检测 |
| [人脸检测](./contrib/FaceBoxes) | python | >=2.0.4 | 对图像中的人脸进行画框并且标注置信度 |
| [口罩识别](./contrib/FaceBoxes) | python | >=2.0.4 | 对原图像的人脸以及口罩进行识别画框 |
| [人脸替换](./contrib/faceswap) | python | >=2.0.4 | 进行人脸检测，脸部关键点推理以及人脸替换，将替换结果可视化并保存 |
| [情绪识别](./contrib/FacialExpressionRecognition) | python | >=2.0.4 | 采集图片中的人脸图像，然后利用情绪识别模型推理情绪类别 |
| [目标跟踪](./contrib/FairMOT) | python | >=2.0.4 | 视频目标检测和跟踪，对行人进行画框和编号 |
| [语义分割](./contrib/FastSCNN) | python | >=2.0.4 | 对图片实现语义分割功能 |
| [疲劳驾驶识别](./contrib/FatigueDrivingRecognition) | python | >=2.0.4 | 对视频中驾驶人员疲劳状态识别与预警 |
| [火灾识别](./contrib/FireDetection) | python | >=2.0.4 | 对视频中高速公路车辆火灾和烟雾的识别告警 |
| [手势关键点](./contrib/GestureKeypointDetection) | python | >=2.0.4 | 检测图像中所有的人手，输出手势关键点连接成的手势骨架 |
| [头部姿态识别](./contrib/HeadPoseEstimation) | python | >=2.0.4 | 对图像中的头部进行姿态识别，输出可视化结果 |
| [安全帽识别](./contrib/HelmetIdentification) | python | >=2.0.4 | 两路视频的安全帽去重识别，并对为佩戴行为告警 |
| [人体语义分割](./contrib/human_segmentation) | c++ | >=2.0.4 | 对输入图片中的人像进行语义分割操作，然后输出mask掩膜图，将其与原图结合，生成标注出人体部分的人体语义分割图片 |
| [个体属性识别](./contrib/Individual) | python | >=2.0.4 | 识别多种人脸属性信息，包括年龄、性别、颜值、情绪、脸型、胡须、发色、是否闭眼、是否配戴眼镜、人脸质量信息及类型等 |
| [语音关键词检测](./contrib/kws) | python | >=2.0.4 | 对语音进行关键词检测 |
| [人像分割](./contrib/MMNET) | python | >=2.0.4 | 基于MMNET解决移动设备上人像抠图的问题，旨在以最小的模型性能降级在移动设备上获得实时推断 |
| [单目深度估计](./contrib/MonocularDepthEstimation) | python | >=2.0.4 | 基于AdaBins室内模型的单目深度估计，输出目标图像的深度图 |
| [多路视频检测](./contrib/MultiChannelVideoDetection) | c++ | >=2.0.4 | 同时对两路本地视频或RTSP视频流(H264或H265)进行YOLOv3目标检测，生成可视化结果 |
| [小麦检测](./contrib/mxBase_wheatDetection) | c++ | >=2.0.4 | 使用yolov5对图像中的小麦进行识别检测 |
| [OCR身份证检测识别](./contrib/OCR/IDCardRecognition) | python | >=2.0.4 | 对身份证进行识别和检测 |
| [OCR关键词检测](./contrib/OCR/KeywordDetection) | python | >=2.0.4 | 对图片进行识别并检测是否包含指定关键词 |
| [人体关键点检测](./contrib/OpenposeKeypointDetection) | python | >=2.0.4 | 输入一幅图像，可以检测得到图像中所有行人的关键点并连接成人体骨架 |
| [行人属性检测](./contrib/PedestrianAttributeRecognition) | python | >=2.0.4 | 对检测图片中行人的定位和属性进行识别 |
| [人群密度计数](./contrib/PersonCount) | python | >=2.0.4 | 输入一幅人群图像，输出图像当中人的计数（估计）的结果 |
| [文本检测](./contrib/PixelLink) | python | >=2.0.4 | 识别图像文本的位置信息，将识别到的文本位置用线条框选出来 |
| [人像分割与背景替换](./contrib/PortraitSegmentation) | python | >=2.0.4 | 使用Portrait模型对输入图片中的人像进行分割，然后与背景图像融合，实现背景替换 |
| [行人重识别](./contrib/ReID) | python | >=2.0.4 | 检索给定照片中的行人ID，并与特征库对比展示 |
| [遥感影像地块分割检测样例](./contrib/RemoteSensingSegmentation) | python | >=2.0.4 | 输出遥感图像的可视化语义分割图 |
| [无人机遥感旋转目标检测](./contrib/RotateObjectDetection) | python | >=2.0.4 | 输入一张待检测图片，可以输出目标旋转角度检测框，并有可视化呈现 |
| [3D目标检测](./contrib/RTM3DTargetDetection) | python | >=2.0.4 | 对道路单色RGB图像进行三维目标检测 |
| [情感极性分类](./contrib/SentimentAnalysis) | python | >=2.0.4 | 输入一段句子，可以判断该句子属于哪个情感极性 |
| [发言者识别](./contrib/SpeakerRecog) | python | >=2.0.4 | 对发言者进行识别。如果声纹库中不包含当前说话人，则对当前说话人进行注册并保存至声纹库，否则给出识别结果 |
| [图像超分辨率](./contrib/SuperResolution) | python | >=2.0.4 | 对输入的图片利用VDSR模型进行超分辨率重建 |
| [车道线检测](./contrib/UltraFastLaneDetection) | python | >=2.0.4 | 对图像中的车道线进行检测，并对检测到的图像中的每一条车道线进行识别 |
| [车流量统计](./contrib/VehicleCounting) | c++ | >=2.0.4 | 对视频中的车辆进行计数，实现对本地视频（H264）进行车辆追踪并计数，最后生成可视化结果 |
| [视频手势识别运行](./contrib/VideoGestureRecognition) | c++ | >=2.0.4 | 对本地视频（H264）进行手势识别并分类，生成可视化结果 |

## 插件使用
**以下表单描述了各插件在哪些样例中有使用，供用户查找参考**   
|             插件名称             |                         参考设计位置                         |                         具体样例名称                         |
| :------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|              appsrc              |                            run包                             |                             通用                             |
|           mxpi_rtspsrc           |                            run包                             | SampleOsdVideo2Channels.pipeline<br>VideoObjectDetection.pipeline<br>SampleMotsimplesortv2.pipeline<br>SampleSkipframe.pipeline |
|        mxpi_dataserialize        |                            run包                             |                             通用                             |
|             appsink              |                            run包                             |                             通用                             |
|             fakesink             |                            run包                             |                             通用                             |
|             filesink             |                            run包                             |                      SampleOsd.pipeline                      |
|       mxpi_parallel2serial       |                            run包                             | SampleNmsoverLapedroiv2.pipeline<br>SampleOsdVideo2Channels.pipeline |
|         mxpi_distributor         |                            run包                             | SampleNmsoverLapedroiv2.pipeline<br/>SampleOsdVideo2Channels.pipeline |
|         mxpi_synchronize         | [gitee-mxVision](https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/mxVision/AllObjectsStructuring) |                    AllObjectsStructuring                     |
|        mxpi_datatransfer         | [gitee-mxVision](https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/mxVision/MultiThread) |                         MultiThread                          |
|       mxpi_nmsoverlapedroi       |                             弃用                             |                                                              |
|      mxpi_nmsoverlapedroiV2      |                            run包                             |               SampleNmsoverLapedroiv2.pipeline               |
|        mxpi_roigenerator         |                            run包                             |              SemanticSegPostProcessor.pipeline               |
|     mxpi_semanticsegstitcher     |                            run包                             |              SemanticSegPostProcessor.pipeline               |
|       mxpi_objectselector        |                            run包                             |                SampleObjectSelector.pipeline                 |
|          mxpi_skipframe          |                            run包                             |                   SampleSkipframe.pipeline                   |
|        mxpi_imagedecoder         |                            run包                             |                             通用                             |
|         mxpi_imageresize         |                            run包                             |                             通用                             |
|          mxpi_imagecrop          |                            run包                             | Sample.pipeline<br>SampleNmsoverLapedroiv2.pipeline<br>SampleObjectSelector.pipeline<br>SampleOsdVideo2Channels.pipeline |
|        mxpi_videodecoder         |                            run包                             |                             通用                             |
|        mxpi_videoencoder         |                            run包                             |               SampleOsdVideo2Channels.pipeline               |
|        mxpi_imageencoder         |                            run包                             |                      SampleOsd.pipeline                      |
|       mxpi_imagenormalize        |                            run包                             |                SampleImageNormalize.pipeline                 |
|      mxpi_opencvcentercrop       |                            run包                             |               SampleOpencvCenterCrop.pipeline                |
|       mxpi_warpperspective       |                    mindxsdk-referenceapps                    |                    GeneralTextRecognition                    |
|          mxpi_rotation           |                    mindxsdk-referenceapps                    |                    GeneralTextRecognition                    |
|         mxpi_modelinfer          |                             弃用                             |                                                              |
|         mxpi_tensorinfer         |                            run包                             |                             通用                             |
|     mxpi_objectpostprocessor     |                            run包                             |      SampleObjectSelector.pipeline<br/>Sample.pipeline       |
|     mxpi_classpostprocessor      |                            run包                             | Sample.pipeline<br>SampleOsdVideo2Channels.pipeline<br>BertMultiPorts.pipeline |
|  mxpi_semanticsegpostprocessor   |                            run包                             |              SemanticSegPostProcessor.pipeline               |
| mxpi_textgenerationpostprocessor |                            run包                             |             TextGenerationPostProcessor.pipeline             |
|   mxpi_textobjectpostprocessor   |                    mindxsdk-referenceapps                    |                    GeneralTextRecognition                    |
|    mxpi_keypointpostprocessor    |                            run包                             |                KeyPointPostProcessor.pipeline                |
|        mxpi_motsimplesort        |                             弃用                             |                                                              |
|       mxpi_motsimplesortV2       |                            run包                             |                                                              |
|        mxpi_facealignment        |                    mindxsdk-referenceapps                    |                    FaceFeatureExtraction                     |
|      mxpi_qualitydetection       | [gitee-mxVision](https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/mxVision/VideoQualityDetection) |                    VideoQualityDetection                     |
|          mxpi_dumpdata           | [用户指南](https://support.huawei.com/enterprise/zh/doc/EDOC1100234263/ba172876) |                                                              |
|          mxpi_loaddata           | [用户指南](https://support.huawei.com/enterprise/zh/doc/EDOC1100234263/ba172876) |                                                              |
|          mxpi_opencvosd          |                            run包                             |                      SampleOsd.pipeline                      |
|     mxpi_object2osdinstances     |                            run包                             |               SampleOsdVideo2Channels.pipeline               |
|     mxpi_class2osdinstances      |                            run包                             |               SampleOsdVideo2Channels.pipeline               |
|      mxpi_osdinstancemerger      |                            run包                             |               SampleOsdVideo2Channels.pipeline               |
|       mxpi_channelselector       |                            run包                             |               SampleOsdVideo2Channels.pipeline               |
|    mxpi_channelimagesstitcher    |                            run包                             |               SampleOsdVideo2Channels.pipeline               |
|  mxpi_channelosdcoordsconverter  |                            run包                             |               SampleOsdVideo2Channels.pipeline               |
|       mxpi-bufferstablizer       |                            run包                             |               SampleOsdVideo2Channels.pipeline               |



## 文档

参考各组件：制造质检 [mxManufacture](https://www.hiascend.com/software/mindx-sdk/mxManufacture/community)、视觉分析 [mxVision](https://www.hiascend.com/software/mindx-sdk/mxVision/community)、检索聚类 [mxIndex](https://www.hiascend.com/software/mindx-sdk/mxIndex/community)内的**用户手册**链接获取相关文档。

## 社区

昇腾社区鼓励开发者多交流，共学习。开发者可以通过以下渠道进行交流和学习。

昇腾社区网站：hiascend.com

昇腾论坛：https://bbs.huaweicloud.com/forum/forum-726-1.html
>SDK专属空间位于**MindX应用使能**子目录下

昇腾官方qq群：965804873

## 贡献代码

欢迎参与贡献。更多详情，请参阅我们的[CONTRIBUTING.md](./contrib/CONTRIBUTING.md)

## 版权说明

请参阅 [License.md](License.md)