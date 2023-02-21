#!/bin/bash
rm -fr build
mkdir -p build
cd build

cmake ..
make -j || {
    ret=$?
    echo "Failed to build."
    exit ${ret}
}

cd ..
./sample
exit 0