## 第一步，编译clipper动态库
- 在[网站](https://sourceforge.net/projects/polyclipping/files/)下载`clipper_ver6.4.2.zip`压缩包，解压后将路径cpp下的 `clipper.hpp、clipper.cpp` 到本地路径src/Clipper下
- 在Clipper目录下新建并进入路径：
  - `mkdir build`
  - `cd build`
- 编译并安装：
  - `cmake ..`
  - `make -j`
  - `make install`
- 可在根目录下的lib中看到编译生成的文件：`libclipper.so`。将lib的绝对路径加入系统环境变量：
  -   `vim ~/.bashrc`
  -   将 ` export LD_LIBRARY_PATH=<Project_Root>/lib:$LD_LIBRARY_PATH ` 加在最后一行（注意修改路径）, 保存退出
  -    `source ~/.bashrc` 使之生效。

## 第二步，编译后处理动态库DBPostProcess
-  进入src/DBPostProcess路径，进行类似第一步的编译流程，创建并进入路径：
   -  `mkdir build`
   -  `cd build`
-  编译安装：
   -  `cmake ..`
   -  `make -j`
   -  `make install`
-  可在src同级路径lib中看到 `libDBPostProcess.so`， 确保pipeline中元件`"mxpi_textobjectpostprocessor0"`的参数`"postProcessLibPath"`为该lib所在的绝对路径。

## 注意事项
- DB后处理目前支持两种缩放方式：拉伸缩放`Resizer_Stretch`、 等比例缩放`Resizer_KeepAspectRatio_Fit`。