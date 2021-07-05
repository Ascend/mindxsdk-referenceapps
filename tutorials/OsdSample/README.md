# SampleOsd 样例说明
* 本样例从ExternalOsdInstances.json构建一个绘图单元集合（MxpiOsdInstancesList）的元数据（metadata）并送入stream
* 上传一张jpg格式图片并重命名为test.jpg，在运行目录下执行run.sh。请勿使用大分辨率图片
* 如构建的proto数据正确则可在程序运行结束后在运行目录找到图片testout.jpg，此图片为test.jpg经过缩放后加上绘图单元集合后的输出结果。