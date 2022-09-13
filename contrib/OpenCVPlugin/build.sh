mkdir OpenCVPlugin/build
cd OpenCVPlugin/build
cmake ..
make -j4
chmod 400 ../lib/plugins/libOpenCVPlugin.so
cp ../lib/plugins/libOpenCVPlugin.so ${MX_SDK_HOME}/lib/plugins
