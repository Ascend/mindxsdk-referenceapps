RTSP_URL="rtsp://192.168.88.109:8554/cut_video/"

cat label.txt|
while read line
do
    sfile=`echo ${line%% *}`
    str=`echo ${line#* }`
    label1=`echo ${str%% *}`
    frame=`echo ${line##* }`
    echo $sfile
    python3.7 test.py \
        --url_video ${RTSP_URL}$sfile \
        --label $label1 \
        --frame_num  $frame
done