# VideoQualityDetection

## 1 简介

VideoQualityDetection基于mxVision SDK开发的参考用例，以昇腾Atlas300卡为主要的硬件平台，用于对IPC视频进行不同的算法功能质量诊断，在日志中记录诊断信息。目前支持的算法功能有：视频亮度检测，视频遮挡检测，视频模糊检测，视频噪声检测，视频偏色检测，视频条纹检测，视频黑屏检测，视频冻结检测，视频抖动检测，视频突变检测，PTZ云台运动检测
## 2 环境依赖

- 支持的硬件形态和操作系统版本

| 硬件形态                             | 操作系统版本   |
| ----------------------------------- | -------------- |
| x86_64+Atlas 300I 推理卡（型号3010） | Ubuntu 18.04.1 |
| x86_64+Atlas 300I 推理卡（型号3010） | CentOS 7.6     |
| ARM+Atlas 300I 推理卡 （型号3000）   | Ubuntu 18.04.1 |
| ARM+Atlas 300I 推理卡 （型号3000）   | CentOS 7.6     |

- 软件依赖

| 软件名称 | 版本   |
| -------- | ------ |
| cmake    | 3.5.1+ |
| mxVision | 0.2    |

## 3 准备

**步骤1：** 参考安装教程《mxVision 用户指南》安装 mxVision SDK。

**步骤2：** 配置 mxVision SDK 环境变量。

`export MX_SDK_HOME=${安装路径}/mxVision `

注：本例中mxVision SDK安装路径为 /root/MindX_SDK。

**步骤3：** 修改项目根目录下 VideoQualityDetection/pipeline/VideoQualityDetection.pipeline文件：

①：将所有“rtspUrl”字段值（"rtsp://xxx.xxx.xxx.xxx:xxxx/xxxx.264"）替换为可用的 rtsp 流源地址（目前只支持264格式的rtsp流，264视频的分辨率范围最小为128*128，最大为4096*4096，不支持本地视频）；

②：将所有“deviceId”字段值替换为实际使用的device的id值，可用的 device id 值可以使用如下命令查看：`npu-smi info`

③：如需配置多路输入视频流，需要配置多个拉流、解码、质量诊断插件。

④：用户可以自定义视频质量检测插件“qualityDetectionConfigContent“字段的值，用例会优先根据该字段的配置作为参数值运行，未配置到的参数会使用默认值，下面是该字段所有参数值的介绍，限制和默认值：  
"FRAME_LIST_LEN"：插件存放视频帧队列长度，当视频帧队列满了之后才开始质量诊断，必须是大于等于2的正整数，下面所有算法的帧间隔须小于该值，默认值“20”  
"BRIGHTNESS_SWITCH"：视频亮度检测算法开关，“true”为进行该功能检测，“false”为不进行该功能检测，默认值为“false”  
"BRIGHTNESS_FRAME_INTERVAL"：视频亮度检测帧间隔，每隔该间隔数就会抽取当前视频帧来进行质量诊断，必须是正整数并且小于FRAME_LIST_LEN，默认值“10”  
"BRIGHTNESS_THRESHOLD"：视频亮度检测算法阈值，用户可根据实际场景进行修改，默认值“1”  
"OCCLUSION_SWITCH"：视频遮挡检测算法开关，“true”为进行该功能检测，“false”为不进行该功能检测，默认值为“false”  
"OCCLUSION_FRAME_INTERVAL"：视频遮挡检测帧间隔，每隔该间隔数就会抽取当前视频帧来进行质量诊断，必须是正整数并且小于FRAME_LIST_LEN，默认值“10” 
"OCCLUSION_THRESHOLD"：视频遮挡检测算法阈值，阈值范围是0-1，用户可根据实际场景进行修改，默认值“0.32”  
"BLUR_SWITCH"：视频模糊检测算法开关，“true”为进行该功能检测，“false”为不进行该功能检测，默认值为“false”  
"BLUR_FRAME_INTERVAL"：视频模糊检测帧间隔，每隔该间隔数就会抽取当前视频帧来进行质量诊断，必须是正整数并且小于FRAME_LIST_LEN，默认值“10”  
"BLUR_THRESHOLD"：视频模糊检测算法阈值，用户可根据实际场景进行修改，默认值“2000”  
"NOISE_SWITCH"：视频噪声检测算法开关，“true”为进行该功能检测，“false”为不进行该功能检测，默认值为“false”  
"NOISE_FRAME_INTERVAL"：视频噪声检测帧间隔，每隔该间隔数就会抽取当前视频帧来进行质量诊断，必须是正整数并且小于FRAME_LIST_LEN，默认值“10”  
"NOISE_THRESHOLD"：视频噪声检测算法阈值，用户可根据实际场景进行修改，默认值“0.005”  
"COLOR_CAST_SWITCH"：视频偏色检测算法开关，“true”为进行该功能检测，“false”为不进行该功能检测，默认值为“false”  
"COLOR_CAST_FRAME_INTERVAL"：视频偏色检测帧间隔，每隔该间隔数就会抽取当前视频帧来进行质量诊断，必须是正整数并且小于FRAME_LIST_LEN，默认值“10”  
"COLOR_CAST_THRESHOLD"：视频偏色检测算法阈值，用户可根据实际场景进行修改，默认值“1.5”  
"STRIPE_SWITCH"：视频条纹检测算法开关，“true”为进行该功能检测，“false”为不进行该功能检测，默认值为“false”  
"STRIPE_FRAME_INTERVAL"：视频条纹检测帧间隔，每隔该间隔数就会抽取当前视频帧来进行质量诊断，必须是正整数并且小于FRAME_LIST_LEN，默认值“10”  
"STRIPE_THRESHOLD"：视频条纹检测算法阈值，用户可根据实际场景进行修改，默认值“0.0015”  
"DARK_SWITCH"：视频黑屏检测算法开关，“true”为进行该功能检测，“false”为不进行该功能检测，默认值为“false”  
"DARK_FRAME_INTERVAL"：视频黑屏检测帧间隔，每隔该间隔数就会抽取当前视频帧来进行质量诊断，必须是正整数并且小于FRAME_LIST_LEN，默认值“10”  
"DARK_THRESHOLD"：视频黑屏检测算法阈值，阈值范围是0-1，用户可根据实际场景进行修改，默认值“0.72”  
"VIDEO_FREEZE_SWITCH"：视频冻结检测算法开关，“true”为进行该功能检测，“false”为不进行该功能检测，默认值为“false”  
"VIDEO_FREEZE_FRAME_INTERVAL"：视频冻结检测帧间隔，每隔该间隔数就会抽取当前视频帧与前一间隔帧来进行质量诊断，必须是正整数并且小于FRAME_LIST_LEN，默认值“10”  
"VIDEO_FREEZE_THRESHOLD"：视频冻结检测算法阈值，阈值范围是0-1，用户可根据实际场景进行修改，默认值“0.1”   
"VIEW_SHAKE_SWITCH"：视频抖动检测算法开关，“true”为进行该功能检测，“false”为不进行该功能检测，默认值为“false”   
"VIEW_SHAKE_FRAME_INTERVAL"：视频抖动检测帧间隔，每隔该间隔数就会抽取当前视频帧与前一间隔帧来进行质量诊断，必须是正整数并且小于FRAME_LIST_LEN，默认值“10”   
"VIEW_SHAKE_THRESHOLD"：视频抖动检测算法阈值，阈值范围是10-100，用户可根据实际场景进行修改，默认值“20”    
"SCENE_MUTATION_SWITCH"：视频突变检测算法开关，“true”为进行该功能检测，“false”为不进行该功能检测，默认值为“false”   
"SCENE_MUTATION_FRAME_INTERVAL"：视频突变检测帧间隔，每隔该间隔数就会抽取当前视频帧与前一间隔帧来进行质量诊断，必须是正整数并且小于FRAME_LIST_LEN，默认值“10”   
"SCENE_MUTATION_THRESHOLD"：视频突变检测算法阈值，阈值范围是0-1，用户可根据实际场景进行修改，默认值“0.5”   
"PTZ_MOVEMENT_SWITCH"：PTZ云台运动检测算法开关，“true”为进行该功能检测，“false”为不进行该功能检测，默认值为“false”   
"PTZ_MOVEMENT_FRAME_INTERVAL"：PTZ云台运动检测帧间隔，每隔该间隔数就会抽取该间隔的视频帧进行质量诊断，必须是大于1的正整数并且小于FRAME_LIST_LEN，默认值“10”  
"PTZ_MOVEMENT_THRESHOLD"：PTZ云台运动检测算法阈值，用户可根据实际场景进行修改，默认值“0.95”   

## 4 运行

运行
`bash run.sh`

正常启动后，控制台会输出视频质量检测结果，结果日志将保存到`${安装路径}/mxVision/logs`中

手动执行ctrl + C结束程序

若中途视频流获取异常（例如视频流中断），程序会等待处理，不会自动退出
