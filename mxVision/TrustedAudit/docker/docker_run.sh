# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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
#!/bin/bash
VER=1.0
NETWORK_NAME="trusted_audit_net"
FILTER_NAME=`docker network ls | grep $NETWORK_NAME | awk '{ print $2 }'`
if [ "$FILTER_NAME" != $NETWORK_NAME ]; then #不存在就创建
    docker network create --subnet=172.18.0.0/16 $NETWORK_NAME
fi
CUR_PATH=$(cd "$(dirname "$0")" || { warn "Failed to check path" ; exit ; } ; pwd)
VPATH=$CUR_PATH #查找docker image是否存在，如果不存在则build image
name="trustedaudit/elasticsearch"
FILTER_NAME=$(docker images | grep $name | awk '{ print $1 }')
FILTER_VER=`docker images | grep $name | awk '{ print $2 }'`
if [[ $FILTER_NAME == $name ]] && [[ $FILTER_VER == $VER ]];  then
    echo "trustedaudit/elasticsearch docker image exist"
else #未查询到docker image，根据dockerfile build对应image 123
    echo "build trustedaudit/elasticsearch docker image"
    docker build -f Dockerfile_es -t trustedaudit/elasticsearch:$VER .
fi
ES_IMAGE_NAME=trustedaudit/elasticsearch:$VER #Create elatsticsearch docker container interface
ES_CONTAINER_NAME=container_elasticsearch_$VER
FILTER_NAME=`docker ps -a | grep $ES_CONTAINER_NAME | awk '{ print $1 }'`
if [ "$FILTER_NAME" == "" ]; then
    echo 'create elasticsearch contianer instance now'
    docker run -d \
	--name=$ES_CONTAINER_NAME \
	--net $NETWORK_NAME \
        --ip 172.18.0.3     \
	-v es-data:/usr/share/elasticsearch/data \
	-v /etc/localtime:/etc/localtime:ro \
        -v /etc/timezone:/etc/timezone:ro \
        -e "discovery.type=single-node" \
	-e ES_JAVA_OPTS="-Xms64m -Xmx256m" \
	-p 9200:9200 \
        $ES_IMAGE_NAME
else
	echo 'elasticsearch contianer instance exists'
fi
name="trustedaudit/opengauss" #查找docker image是否存在，如果不存在则build image
FILTER_NAME=$(docker images | grep $name | awk '{ print $1 }')
FILTER_VER=`docker images | grep $name | awk '{ print $2 }'`
if [[ $FILTER_NAME == $name ]] && [[ $FILTER_VER == $VER ]];  then
    echo "trustedaudit/opengauss docker image exist"
else #未查询到docker image，根据dockerfile build对应image
    echo "build trustedaudit/opengauss docker image"
    docker build -f Dockerfile_opengauss -t trustedaudit/opengauss:$VER .
fi
DB_PWD=Enmo@123 #Create gauss docker container interface
IMAGE_NAME=trustedaudit/opengauss:$VER
CONTAINER_NAME=container_gauss_$VER
FILTER_NAME=`docker ps -a | grep $CONTAINER_NAME | awk '{ print $7 }'`
if [ "$FILTER_NAME" == "" ]; then
     echo 'create gaussDB container instance now'
     docker run -d \
        --name $CONTAINER_NAME \
        --net $NETWORK_NAME \
        --ip 172.18.0.11     \
        --privileged=true \
        -v /etc/localtime:/etc/localtime:ro \
        -v /etc/timezone:/etc/timezone:ro \
        -e GS_PASSWORD=$DB_PWD \
        -p 5432:5432 \
        $IMAGE_NAME
else
	echo 'gauss contianer instance exists'
fi
name="trustedaudit/python" #查找docker image是否存在，如果不存在则build image
FILTER_NAME=$(docker images | grep $name | awk '{ print $1 }')
FILTER_VER=`docker images | grep $name | awk '{ print $2 }'`
if [[ $FILTER_NAME == $name ]] && [[ $FILTER_VER == $VER ]];  then
    echo "trustedaudit/python docker image exist"
else
    echo "build trustedaudit/python docker image" #未查询到docker image，根据dockerfile build对应image
    docker build -f Dockerfile_python -t trustedaudit/python:$VER .
fi
PYTHON_IMAGE_NAME=trustedaudit/python:$VER
PYTHON_CONTAINER_NAME=container_python_$VER
FILTER_NAME=`docker ps -a | grep $PYTHON_CONTAINER_NAME | awk '{ print $1 }'`
if [ "$FILTER_NAME" == "" ]; then
     echo 'create python contianer instance now'
     docker run -d -t \
	--name=$PYTHON_CONTAINER_NAME \
	--net $NETWORK_NAME \
        --ip 172.18.0.4     \
	-v $VPATH/../trusted_audit/src:/home/ \
	-v /etc/localtime:/etc/localtime:ro \
        -v /etc/timezone:/etc/timezone:ro \
	-p 1234:1234 \
        $PYTHON_IMAGE_NAME
else
	echo 'python contianer instance exists'
fi
docker start container_elasticsearch_$VER_$VER
docker start container_gauss_$VER
docker start container_python_$VER
sleep 20s
docker exec container_python_$VER python -u /home/database_init.py >> /tmp/database_init.log 2>&1 &