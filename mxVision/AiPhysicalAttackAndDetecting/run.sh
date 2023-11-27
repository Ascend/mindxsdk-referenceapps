#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
# Description: Start script.
# Author: MindX SDK
# Create: 2023
# History: NA

set -e

usage() {
  cat << EOF
Usage: ./run.sh [options]

Options:
  -h/--help        display this help and exit
  -i/--host        listening ip address.
  -p/--port        listening port. The default value is 8888
  -d/--device_id   choose device id. The default value 0.
EOF
}

unset host
unset port
unset device_id

check_params()
{
    if [ "$1" == "" ]; then
        echo "ERROR: please set value!"
        usage
        exit 1
    fi
}

while [ "$1" != "" ]; do
    case "$1" in
        -i | --host )            shift
                                 host="$1"
                                 ;;
        -p | --port )            shift
                                 port="$1"
                                 ;;
        -d | --device_id )       shift
                                 device_id="$1"
                                 ;;
        -h | --help )            usage
                                 exit
                                 ;;
        * )                      usage
                                 exit 1
                                 ;;
    esac
    check_params $1
    shift
done

unset arguments
if [ -z "$host" ]; then
    echo "Ip is none, please set the server ip address."
    exit -1
fi
arguments="${arguments} -i ${host}"

if [ "$port" ]; then
    arguments="${arguments} -p ${port}"
fi
if [ "$device_id" ]; then
    arguments="${arguments} -d ${device_id}"
fi

python3 app.py ${arguments}
