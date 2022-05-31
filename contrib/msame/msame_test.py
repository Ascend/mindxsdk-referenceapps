# -*- coding: utf-8 -*- 
# Copyright 2020 Huawei Technologies Co., Ltd
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
# ============================================================================
import time
import os
import sys
import stat
import argparse
import numpy as np
import mindx.sdk as sdk


def parse_args():
    parser = argparse.ArgumentParser(description='msame-python')
    parser.add_argument('--input', required=False, type=str, default='')
    parser.add_argument('--model', required=True, type=str, help='model is necessary')
    parser.add_argument('--output', required=True, type=str, help='check out your output path')
    parser.add_argument('--outfmt', required=True, type=str, help='your output format must in "TXT"  or   "BIN"')
    parser.add_argument('--loop', required=False, type=int, default=1)
    parser.add_argument('--device', required=False, type=int, default=0)

    my_args = parser.parse_args()
    return my_args


type_map = {}
type_map['dtype.uint8'] = np.uint8
type_map['dtype.float32'] = np.float32
type_map['dtype.float16'] = np.float16
type_map['dtype.int8'] = np.int8
type_map['dtype.int32'] = np.int32
type_map['dtype.int16'] = np.int16
type_map['dtype.uint16'] = np.uint16
type_map['dtype.uint32'] = np.uint32
type_map['dtype.int64'] = np.int64
type_map['dtype.uint64'] = np.uint64
type_map['dtype.double'] = np.double

args = parse_args()
loop = args.loop
filepath = args.model
output = args.output
datatype = args.outfmt
device_id = args.device
output = args.output
types_output = []

def infer(saves):
    if not os.path.exists(output):
        os.makedirs(output)
    m = sdk.model(filepath, device_id)
    index = 0
    for i in m.output_dtype:
        types_output.append([])
        types_output[index].append(str(i))
        index += 1
    types_input = str(m.input_dtype[0])
    multi = 1
    f = []
    if len(m.input_shape) == 1:
        if os.path.isdir(args.input):
            isf = 0
            for fi in os.listdir(args.input):
                isf += 1
                try:
                    t = get_t(args.input+'/'+fi, m, type_map[types_input])
                except KeyError:
                    print("KeyError")
                path = fi.split('.')[0]
                one_time = t_save(path, m, t, saves)
            if isf == 0:
                print("It's an empty folder, please check your input")
                sys.exit(0)
        elif args.input == '':
            try:
                t = get_input_num(m, type_map[types_input])
            except KeyError:
                print("KeyError")
            path = filepath
            one_time = t_save(path, m, t, saves)
        else:
            if  not os.path.isfile(args.input):
                print("please check your file!")
                sys.exit(0)
            try:
                t = get_t(args.input, m, type_map[types_input])
            except KeyError:
                print("KeyError")
            if len(args.input.split('/')) == 1:
                path = args.input
            else:
                path = args.input.split('/')[-1].split('.')[0]
            one_time = t_save(path, m, t, saves)
    else:
        try:
            t = get_input_num(m, type_map[types_input])
        except KeyError:
            print("KeyError")
        path = filepath
        one_time = t_save(path, m, t, saves)
    return one_time

def get_input_num(m, input_type):
    inputsize = []
    index = 0
    for j in m.input_shape:
        inputsize.append(1)
        for k in j:
            inputsize[index] *= k
        index += 1
    types_input = str(m.input_dtype[0])
    t = []
    files_name = []
    tis = []
    if args.input == '':
        for i in inputsize:
            try:
                t.append(np.zeros(i, type_map[types_input]))
            except KeyError:
                print("KeyError")
        tis.append(t)
    else:
        binfile = args.input
        all_names = []
        if len(binfile.split(',')) > 1:
            multi_bin = binfile.split(',')
            for i in multi_bin:
                files_name.append(i)
                all_names.append(get_files(i))
        else:
            files_name.append(binfile)
            all_names.append(get_files(binfile))
            if len(all_names[0]) == 0:
                print("It's an empty folder, please check your input")
                sys.exit(0)
        for mul in files_name:
            if all_names[0][0].split('.')[1] == "bin":
                t.append(get_bins(mul, input_type))
            elif all_names[0][0].split('.')[1] == "npy":
                t.append(get_npy(mul, input_type))
        tis = t
    return tis

def t_save(path, m, t, saves):
    multi = 1
    for p in m.input_shape[0]:
        multi = multi * p
    if t[0][0].shape[0] != multi :
        print("Error : Please check the input shape and input dtype")
        sys.exit()
    if len(m.input_shape) == 1:
        tim = sdk.Tensor(t[0][0])
        tim.to_device(0)
    else:
        if args.input == '':
            tim = []
            for bs in t[0]:
                bs = sdk.Tensor(bs)
                bs.to_device(0)
                tim.append(bs)
        else:
            tim = []
            for bs in t:
                bs = sdk.Tensor(bs[0])
                bs.to_device(0)
                tim.append(bs)
    last_time = time.time()
    outputs = m.infer(tim)
    now_time = time.time()
    one_times = now_time-last_time
    outputs[0].to_host()
    nums, shape = get_nums(outputs, types_output)
    if saves:
        save_files(path, outputs, output, datatype, nums, shape, types_output)
    return one_times

def get_t(name, m, input_type):
    ti = []
    if name.split('.')[1] == 'bin':
        ti.append(get_bins(name, input_type))
    elif name.split('.')[1] == 'npy':
        ti.append(get_npy(name, input_type))
    return ti

def get_bins(f, input_type):
    files_bin = []
    bins = []
    if os.path.isdir(f):
        for s in os.listdir(f):
            files_bin.append(np.fromfile(f+"/"+s, dtype=input_type).flatten())
    else:
        files_bin.append(np.fromfile(f, dtype=input_type).flatten())
    return files_bin

def get_npy(f, input_type):
    files_npy = []
    bins = []
    if os.path.isdir(f):
        for s in os.listdir(f):
            files_npy.append(np.load(f+"/"+s).flatten())
    else:
        files_npy.append(np.load(f).flatten())
    return files_npy


def get_files(binfile):
    all_name = []
    if os.path.isdir(binfile):
        for s in os.listdir(binfile):
            all_name.append(s)
    elif os.path.isfile(binfile):
        all_name.append(binfile)
    return all_name


def get_nums(outputs, types_output):
    index = 0
    num_list = []
    shape = []
    #get shape

    for ij in outputs:
        ij.to_host()
        try:
            num = np.array(ij).astype(type_map[types_output[index][0]])
        except KeyError:
            print("KeyError")
        num = num.flatten()
        try:
            num.dtype = type_map[types_output[index][0]]
        except KeyError:
            print("KeyError")
        shape.append(ij.shape[-1])
        num_list.append(num)
        index += 1
    #to list
    i_index = 0
    nums = []
    for i in num_list:
        output_desc = len(i)
        nums.append([])
        for j in range(int(output_desc/shape[i_index])):
            for k in range(j*shape[i_index], (j+1)*shape[i_index]):
                nums[i_index].append(num_list[i_index][k])
        i_index += 1
    return (nums, shape)


def save_files(filepath, outputs, output, datatype, nums, shape, types_output):
    #TXT
    if datatype == 'TXT' or 'txt':
        i_index = 0
        for ik in nums:
            f = os.open(output+'/'+filepath.split('.')[0]+'_'+str(i_index)+".txt", 
                    os.O_RDWR | os.O_APPEND | os.O_CREAT, stat.S_IRWXU)
            os.chmod(output+'/'+filepath.split('.')[0]+'_'+str(i_index)+".txt", stat.S_IRWXU)
            output_desc = len(ik)
            for j in range(int(output_desc/shape[i_index])):
                for k in range(j*shape[i_index], (j+1)*shape[i_index]):
                    os.write(f, str.encode(str(nums[i_index][k])+' '))
                os.write(f, str.encode('\n'))
            i_index += 1
    else:
    #BIN
        i_index = 0
        for i in outputs:
            i.to_host()
            num = np.array(i)
            num = num.flatten()
            num.tofile(output+'/'+filepath.split('.')[0]+'_'+str(i_index)+".bin")
            i_index += 1
    

def get_array(files_bin, input_type):
    new_files = []
    #make new bin
    a = len(files_bin)
    for im in range(a):
        if im == 0:
            new_files.append(files_bin[0])
            continue
        else:
            a = files_bin[im]
            b = new_files[im-1]
            new_files.append(np.concatenate((a, b), axis=0))
    bins = np.array(new_files[-1]).astype(input_type)
    return bins



if  __name__ == '__main__':
    TIMES = 0.0
    TRANS = 1000
    SUM = 0
    SAVES = True
    for mj in range(loop):
        nowtimes = time.time()
        TIMES = infer(SAVES)
        SUM += TIMES
        SAVES = False
        print("loop {0} : Inference time: {1:f} ms".format(mj, TIMES*TRANS))
    print("infer success!")
    print("Inference average time: {0:f} ms".format(SUM/loop*TRANS))


