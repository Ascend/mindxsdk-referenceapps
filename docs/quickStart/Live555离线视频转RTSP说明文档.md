# Live555离线视频转RTSP说明文档

## 1.下载安装包

[http://www.live555.com/liveMedia/public/live555-latest.tar.gz](https://bbs.huaweicloud.com/forum/thread-68720-1-1.html#)

## 2. 解压

执行命令：

```
tar -zxvf live555-latest.tar.gz

cd live/
```

## 3. 编译并安装

执行命令：

```
./genMakefiles linux  #注意后面这个参数是根据当前文件夹下config.<后缀>获取得到的,与服务器架构等有关。

make
```

最后就会在当前目录下生成mediaServer 文件夹，有一个live555MediaServer可执行文件

## 4. 运行

执行命令（所有的视频文件放在mediaServer文件夹同一目录下），产生的RTSP流的地址如下图所示，文件名为上一步放入mediaServer 文件夹的视频文件。

```
cd mediaServer

./live555MediaServer
```


[img](img/20210720145058139.png)
其中rtsp_Url的格式是 rtsp:://host:port/Data，host:port/路径映射到mediaServer/目录下，Data为视频文件的路径。

# 补充

## 1. 视频循环推流

按照以下提示修改文件可以使自主搭建的rtsp循环推流，如果不作更改，则为有限的视频流

在liveMedia库下的ByteStreamFileSource.cpp文件中的95行，找到

```
void ByteStreamFileSource::doGetNextFrame() {

if (feof(fFid) || ferror(fFid) || (fLimitNumBytesToStream && fNumBytesToStream == 0))
{
    handleClosure();
    return;
 }
```

更改为

```
void ByteStreamFileSource::doGetNextFrame() {

if (feof(fFid) || ferror(fFid) || (fLimitNumBytesToStream && fNumBytesToStream == 0)) {
    //handleClosure();**
    //return;**
    fseek(fFid, 0, SEEK_SET);
 }
```

## 2. 高分辨率帧花屏

在使用Live555进行拉流时，出现”**The input frame data was too large for our buffer**“问题，导致丢帧。

解决办法：

在mediaServer的DynamicRTSPServer.cpp文件，修改每一处OutPacketBuffer::maxSize的值，目前更改到800000。



### **源代码修改后都需要重新编译并安装live555.**

```
./genMakefiles <os-platform>
make
```