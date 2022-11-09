# !/usr/bin/env python
# coding=utf-8

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

'''
4.noDet.py:过滤没有检测到目标的图片文件
'''

import sys
import os
import glob

'''
validate_voc_PATH:验证集的voc数据
sdk_predict_voc_PATH：sdk预测的voc数据
检查是否有的数据是无目标的
'''

def equals(varo, vart):

    return varo == vart

def equalZero(veroi):

    return veroi == 0

def error(msg,other="",exit_flag=False):
    if (not msg):
        return
    else:
        print(msg,other)
        if (exit_flag):
            sys.exit()

cur_path = os.path.abspath(os.path.dirname(__file__))
validate_voc_PATH = os.path.join(cur_path, 'det_val_data', 'det_val_voc').replace('\\', '/')
sdk_predict_voc_PATH = os.path.join(cur_path, 'det_val_data', 'det_sdk_voc').replace('\\', '/')


backup_folder = 'backup_no_matches_found'

os.chdir(validate_voc_PATH)
validate_voc_files = glob.glob('*.txt')
if equalZero(len(validate_voc_files)):
    error("Error: no .txt files found in", validate_voc_PATH, exit_flag = True)

os.chdir(sdk_predict_voc_PATH)
sdk_predict_voc_files = glob.glob('*.txt')

if equalZero(sdk_predict_voc_files):
    error("Error: no .txt files found in", sdk_predict_voc_PATH, exit_flag = True)

validate_voc_files = set(validate_voc_files)
sdk_predict_voc_files = set(sdk_predict_voc_files)
print('total ground-truth files:', len(validate_voc_files))
print('total detection-results files: '+ str(len(sdk_predict_voc_files))+'\n')

validate_voc_backup = validate_voc_files - sdk_predict_voc_files
sdk_predict_voc_backup = sdk_predict_voc_files - validate_voc_files

# validate_voc

if not validate_voc_backup:
    error('No backup required for', validate_voc_PATH, exit_flag = False)
else:
    os.chdir(validate_voc_PATH)
    ## create the backup dir if it doesn't exist already
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    for file in validate_voc_backup:
        os.rename(file, backup_folder + '/' + file)

# sdk_predict_voc
if not sdk_predict_voc_backup:
    error('No backup required for', sdk_predict_voc_PATH, exit_flag = False)
else:
    os.chdir(sdk_predict_voc_PATH)
    ## create the backup dir if it doesn't exist already
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    for file in sdk_predict_voc_backup:
        os.rename(file, backup_folder + '/' + file)

msg = "msg"
if validate_voc_backup:
    msg='total ground-truth backup files:'
    print(msg+" "+str(len(validate_voc_backup)))
if sdk_predict_voc_backup:
    msg='total detection-results backup files:'
    print(msg+" "+ str(len(sdk_predict_voc_backup)))

intersection_files = validate_voc_files & sdk_predict_voc_files

msg='total intersected files:'
print(msg, len(intersection_files))
msg="Intersection completed!"
print(msg)
