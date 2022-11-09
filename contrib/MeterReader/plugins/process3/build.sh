# 编译mindx插件，运行环境：华为实验室服务器


# 使用cmake编译插件
mkdir build
mkdir lib
cd lib
mkdir plugins
cd ..

cd build
cmake ..
make
cd ..

# 得到的插件位于../lib/plugins
# 复制到lib中
rm -f "$MX_SDK_HOME/lib/plugins/libmxpi_sampleplugin.so"
# cp lib/plugins/libmxpi_sampleplugin.so /home/wangyi4/MindX_SDK/mxVision-2.0.4/lib/plugins
# chmod 440 /home/wangyi4/MindX_SDK/mxVision-2.0.4/lib/plugins/libmxpi_sampleplugin.so
cp lib/plugins/libmxpi_sampleplugin.so "$MX_SDK_HOME/lib/plugins"
chmod 440 "$MX_SDK_HOME/lib/plugins/libmxpi_sampleplugin.so"



