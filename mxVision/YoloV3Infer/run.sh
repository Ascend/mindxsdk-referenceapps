path_cur=$(dirname $0)

cd $path_cur

rm -rf build
mkdir -p build
cd build

cmake ..
make -j || {
    ret=$?
    echo "Failed to build"
    exit ${ret}
}

cd ..
./mxbaseV2_sample test.jpg # test.jpg could be changed!
exit 0