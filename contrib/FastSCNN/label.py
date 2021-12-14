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
# shutilģ����Ҫ�����ڿ����ļ�
 
# ȡ�õ�ǰĿ¼�µ��ļ������б�
files_list = os.listdir()
# ȡ��python�ű�������
# __file__��ȡ�õ�ǰ�ű�·��,���·���ǡ�\anaconda3\python�������ĸ�ʽ����Ҫʹ�á�\\�����з�
py_name = __file__.split('/')[-1]
 
for file in files_list:
    # ������ļ��ǵ�ǰִ�е�py�ű���������
    if file == py_name:
        continue
    # �����ǰ�ļ���ʽ����һ���ļ��硰.����������
    if not os.path.isfile(file):
        continue
 
    # ȡ�õ�ǰ�ļ����Ƶĸ�ʽ�����з��ļ�����ȡ�����б�Ԫ�أ�
    file_type = file.split('_')[5]
    # ���û��ĳ����ʽ���ļ��У��򴴽�����ļ���
    if not os.path.exists(file_type):
        os.mkdir(file_type)
 
    # ��ȡ��ǰ·��
    path = os.getcwd()
    # ��ȡ�����ļ���·��
    subdir = os.path.join(path,'%s'%file_type)
    # ��������ļ���
    os.chdir(subdir)
    if os.path.exists(file):
        # ����ļ��д��ڵ�ǰ�ļ���������
        continue
    else:
        # ����֮ǰ�ļ��н��й���
        os.chdir(path)
        # shutil.move(Դ�ļ���ָ��·��):�ݹ��ƶ�һ���ļ�
        shutil.move(file,file_type)