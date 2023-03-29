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

func() {
    echo "Usage:"
    echo "bash run.sh -m model_path -c model_config_path -l model_label_path -i image_path [-y]"
    echo "Description:"
    echo "-m model path"
    echo "-c model config file path"
    echo "-l label path for model"
    echo "-i image path to infer"
    echo "-y [Optional] use yuv model, default is not yuv"
    exit -1
}

is_yuv=0
argc=5
args=0
while getopts "i:m:c:l:yh" arg
do
    if [ "$args" -gt "$argc" ]; then
      echo "Error: Wrong usage, too many arguments."
      func
      exit 1
    fi
    case "$arg" in
        i)
        img_path="$OPTARG"
        ;;
        m)
        model_path="$OPTARG"
        ;;
        c)
        model_config_path="$OPTARG"
        ;;
        l)
        model_label_path="$OPTARG"
        ;;
        y)
        is_yuv=1
        ;;
        h)
        func
        exit 1
        ;;
        ?)
        echo "Error: Wrong usage, unknown argument."
        func
        exit 1
        ;;
    esac
    args=$(($args+1))
done

if [ ! -n "$model_path" ]; then
    echo "Error: Required argument \"-m model_path\" is missing." 
    func
    exit 1
fi
if [ ! -n "$model_config_path" ]; then
    echo "Error: Required argument \"-c model_config_path\" is missing." 
    func
    exit 1
fi
if [ ! -n "$model_label_path" ]; then
    echo "Error: Required argument \"-l model_label_path\" is missing." 
    func
    exit 1
fi
if [ ! -n "$img_path" ]; then
    echo "Error: Required argument \"-i img_path\" is missing." 
    func
    exit 1
fi
cd ..
if [ "$is_yuv" -gt 0 ]; then
    ./sample -m "$model_path" -c "$model_config_path" -l "$model_label_path" -i "$img_path" -y
else
    ./sample -m "$model_path" -c "$model_config_path" -l "$model_label_path" -i "$img_path"
fi
exit 0