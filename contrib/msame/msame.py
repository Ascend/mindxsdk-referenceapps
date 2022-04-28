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


def infer(saves):
    filepath = args.model
    output = args.output
    datatype = args.outfmt
    device_id = args.device
    output = args.output
    if not os.path.exists(output):
        os.makedirs(output)

    m = sdk.model(filepath, device_id)
    types_output = []
    index = 0
    for i in m.output_dtype:
        types_output.append([])
        types_output[index].append(str(i))
        index += 1
    types_input = str(m.input_dtype[0])
    t = get_input_num(m) 
    b = len(t)
    for i in range(b):
        t[i] = sdk.Tensor(t[i])
        t[i].to_device(0)
    last_time = time.time()
    outputs = m.infer(t)

    outputs[0].to_host()


    now_time = time.time()
    one_times = now_time-last_time
    nums, shape = get_nums(outputs, types_output) 
    if saves:
        save_files(filepath, outputs, output, datatype, nums, shape, types_output)                                            
    return one_times


def get_input_num(m):
    inputsize = []
    index = 0
    for j in m.input_shape:
        inputsize.append(1)
        for k in j:
            inputsize[index] *= k
        index += 1
    types_input = str(m.input_dtype[0])
    t = []
    if args.input == '':
        for i in inputsize:
            try:
                t.append(np.zeros(i, type_map[types_input]))
            except KeyError:
                print("KeyError")
    else:
        binfile = args.input
        if len(binfile.split(',')) > 1:
            multi_bin = binfile.split(',')
            for i in multi_bin:
                try:
                    if multi_bin.split('.')[1] == 'bin':
                        t.append(get_array(i, type_map[types_input]))
                    else:
                        t.append(get_npy(i, type_map[types_input]))
                except KeyError:
                    print("KeyError")
        else:
            if len(binfile.split('.')) == 2:
                try:
                    if binfile.split('.')[1] == 'bin':
                        t.append(get_array(binfile, type_map[types_input]))
                    else:
                        t.append(get_npy(binfile, type_map[types_input]))
                except KeyError:
                    print("KeyError")
    return t
 

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
    filepath = filepath.split('/')[-1]
    #TXT
    if datatype == 'TXT':
        i_index = 0
        for ik in nums:
            f = os.open(output+'/'+filepath.split('.')[0]+'_'+str(i_index)+".txt", 
                    os.O_RDWR | os.O_APPEND | os.O_CREAT,stat.S_IRWXU)
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
    

def get_array(binfile, input_type):
    files_bin = []
    #make list  files_bin
    if os.path.isdir(binfile):
        for s in os.listdir(binfile):
            files_bin.append(np.fromfile(binfile+"/"+s, dtype=input_type).flatten())
    elif os.path.isfile(binfile):
        files_bin.append(np.fromfile(binfile, dtype=input_type).flatten())
    new_files = []
    #make new bin
    a = len(files_bin)-1+1
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


def get_npy(binfile, input_type):
    files_bin = []
    #make list  files_bin
    if os.path.isdir(binfile):
        for s in os.listdir(binfile):
            files_bin.append(np.load(binfile+"/"+s).flatten())
    elif os.path.isfile(binfile):
        files_bin.append(np.load(binfile).flatten())
    new_file = []
    #make new bin
    a = len(files_bin)-1+1
    for im in range(a):
        if im == 0:
            new_files.append(files_bin[0])
            continue
        else:
            a = files_bin[im]
            b = new_file[im-1]
            new_file.append(np.concatenate((a, b), axis=0))
    bins = np.array(new_files[-1]).astype(input_type)
    return bins


if  __name__ == '__main__':
    times = 0.0
    saves = True
    for mj in range(loop):
        nowtimes = time.time()
        times = infer(saves)
        saves = False
        times += times
        print("loop {0} : Inference time: {1:f} ms".format(mj, times*1000))
    print("infer success!")
    print("Inference average time: {0:f} ms".format(times/loop*1000))


