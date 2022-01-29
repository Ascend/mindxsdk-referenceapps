#!/bin/bash

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Simple log helper functions
info() { echo -e "\033[1;34m[INFO ][Depend  ] $1\033[1;37m" ; }
warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }

#Build
fileName="zlib"
packageFQDN="zlib@1.2.11-h2"
packageName="zlib"
cd "$fileName" || {
  warn "cd to ./opensource/$fileName failed"
  exit 254
}

info "Building dependency $packageFQDN."
chmod u+x configure
export LDFLAGS="-Wl,-z,noexecstack,-z,relro,-z,now,-s"
export CFLAGS="-fPIE -fstack-protector-all -fPIC -Wall -D_GLIBCXX_USE_CXX11_ABI=0"
export CPPFLAGS="-fPIE -fstack-protector-all -fPIC -Wall -D_GLIBCXX_USE_CXX11_ABI=0"
export CC=aarch64-linux-gnu-gcc
./configure \
  --prefix="$(pwd)/../tmp/$packageName" \
  --shared || {
  warn "Build $packageFQDN failed during autogen"
  exit 254
}

make -s -j || {
  warn "Build $packageFQDN failed during make"
  exit 254
}

make install -j || {
  warn "Build $packageFQDN failed during install"
  exit 254
}

cd ..
info "Build $packageFQDN done."