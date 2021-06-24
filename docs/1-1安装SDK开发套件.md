# 1.1 安装MindX SDK开发套件

## 1.1.1 **前提条件**

- 请参考《用户指南》完成安装前准备。

- 安装环境中已安装推理卡驱动和固件。

- 已经通过获取软件包获取MindX SDK开发套件软件包。

- 将当前用户加入HwHiAiUser用户组获取执行权限。

>在SSH终端中输入以下命令执行
```bash
    npu-smi info
```
>若返回npu设备占用情况表格则表示已获取权限，参考下图。  
![image.png](img/1623207353906.png 'image.png')  
  
>若提示报错则表示权限缺失，下图为可能的样例。  
![image.png](img/1623207817848.png 'image.png')  
>此时需切换到root用户下，执行以下命令将开发用户添加到HwHiAiUser用户组，username为当前用户名，请自行替换。
```bash
  usermod -a -G HwHiAiUser username
```

- “~/.bashrc”文件中LD_LIBRARY_PATH环境变量已包含以下路径（acllib库）：
```bash
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/ascend-toolkit/:${LD_LIBRARY_PATH}"
```

  >latest为用户安装的CANN开发套件包中acllib库所在路径，请用户自行替换。
> 如果运行环境是Atlas 500智能边缘小站，则需要包含以下路径：
```
export LD_LIBRARY_PATH="/home/data/miniD/driver/lib64:${LD_LIBRARY_PATH}"
```
> 修改“~/.bashrc”后需要刷新环境变量，执行：
```
source ~/.bashrc
```
## 1.1.2 **安装步骤**

> 说明：
>
> - {version}为开发套件包版本号，{arch}为操作系统架构，请用户自行替换。
> - 安装路径中不能有空格

**步骤1**  以软件包的安装用户身份SSH登录安装环境。

**步骤2**  将MindX SDK开发套件包上传到安装环境的任意路径下（如：“/home/package”）并用cd命令进入套件包所在路径。

**步骤3**  增加对套件包的可执行权限：            

```
chmod +x Ascend-mindxsdk-mxvision_{version}_linux-{arch}.run
```

**步骤4**  执行如下命令，校验套件包的一致性和完整性：

```
./Ascend-mindxsdk-mxvision_{version}_linux-{arch}.run --check
```

​            若显示如下信息，说明套件包满足一致性和完整性：

```
Verifying archive integrity... 100% SHA256 checksums are OK. All good.
```

**步骤5**  创建MindX SDK开发套件包的安装路径。

- 若用户想指定安装路径，需要先创建安装路径。以安装路径“/home/work/MindX_SDK”为例：

  ```
  mkdir -p /home/work/MindX_SDK
  ```

- 若用户未指定安装路径，软件会默认安装到MindX SDK开发套件包所在的路径。

**步骤6**  安装MindX SDK开发套件包。

- 若用户指定了安装路径。以安装路径“/home/work/MindX_SDK”为例：

  ```
  ./Ascend-mindxsdk-mxvision_{version}_linux-{arch}.run --install --install-path=/home/work/MindX_SDK
  ```

- 若用户未指定安装路径：

  ```
  ./Ascend-mindxsdk-mxvision_{version}_linux-{arch}.run --install
  ```

   安装完成后，若未出现错误信息，表示软件成功安装于指定或默认路径下：
```
Uncompressing ASCEND MINDXSDK RNN PACKAGE 100%
```

**步骤7**  环境变量生效。

​            在当前窗口手动执行以下命令，让MindX SDK的环境变量生效。

```
source ~/.bashrc
```
