mkdir OpenCVPlugin/src/OpenCVPlugin/build
cd OpenCVPlugin/src/OpenCVPlugin/build
cmake ..
make -j4
chmod 400 ../lib/plugins/libmxpi_OpenCVPlugin.so
cp ../lib/plugins/libmxpi_OpenCVPlugin.so ${MX_SDK_HOME}/lib/plugins
