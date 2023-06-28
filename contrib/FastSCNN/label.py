"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import shutil

files_list = os.listdir()
py_name = __file__.split('/')[-1]
 
for file in files_list:
    if file == py_name:
        continue
    if not os.path.isfile(file):
        continue
 
    file_type = file.split('_')[4]
    if not os.path.exists(file_type):
        os.mkdir(file_type)
 
    path = os.getcwd()
    subdir = os.path.join(path, '%s' % file_type)
    os.chdir(subdir)
    if os.path.exists(file):
        continue
    else:
        os.chdir(path)
        shutil.move(file, file_type)