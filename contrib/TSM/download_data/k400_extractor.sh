# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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
# Download directories vars
#!/bin/bash

root_dl="../data"
root_dl_targz="."

# Make root directories
[ ! -d $root_dl ] && mkdir $root_dl

# Extract train
curr_dl=$root_dl_targz
curr_extract=$root_dl
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
    [[ $f == *.tar ]] && rm -rf $curr_dl/$f
done

# Extraction complete
echo -e "\nExtractions complete!"