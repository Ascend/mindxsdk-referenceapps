mkdir OpenCVPlugin/build
cd OpenCVPlugin/build
cmake ..
make -j4
cp ../lib/plugins/libOpenCVPlugin.so ${MX_SDK_HOME}/lib/plugins
