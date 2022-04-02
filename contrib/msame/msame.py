import mindx.sdk as sdk
import time
import numpy as np
import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='msame-python')
    parser.add_argument('--input',required=False,type=str,default='')
    parser.add_argument('--model',required=True,type=str,help='model is necessary')
    parser.add_argument('--output',required=True,type=str,help='check out your output path')
    parser.add_argument('--outfmt',required=True,type=str,help='your output format must in "TXT"  or   "BIN"')
    parser.add_argument('--loop',required=False,type=int,default=1)
    parser.add_argument('--device',required=False,type=int,default=0)

    args = parser.parse_args()
    return args

def make_typemap():
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
    return type_map

args = parse_args()
type_map = make_typemap()
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
    print(m.input_dtype)
    print(m.input_shape)
    print(m.output_dtype)
    types_output = []
    index = 0
    for i in m.output_dtype:
        types_output.append([])
        types_output[index].append(str(i))
        index+=1
    #types_output = str(m.output_dtype)
    types_input = str(m.input_dtype[0])
    t = get_input_num(m) 

    print(t)

    for i in range(len(t)):
        t[i] = sdk.Tensor(t[i])
        t[i].to_device(0)
    last_time = time.time()
    outputs = m.infer(t)

    outputs[0].to_host()
    print(np.array(outputs[0]))


    now_time = time.time()
    times = now_time-last_time
    nums,shape = get_nums(outputs,types_output) 
    if saves:
        save_files(filepath,outputs,output,datatype,nums,shape,types_output)
                                                
    return times
def get_input_num(m):
    inputsize=[]
    index = 0
    for j in m.input_shape:
        inputsize.append(1)
        for i in j:
            inputsize[index]*=i
        index += 1
    types_input = str(m.input_dtype[0])
    t = []
    if args.input=='':
        for i in range(len(inputsize)):
            t.append(np.zeros(inputsize[i],type_map[types_input]))
    else:
        binfile = args.input
        if len(binfile.split(',')) > 1:
            multi_bin = binfile.split(',')
            for i in range(len(multi_bin)):
                t.append(get_array(multi_bin[i],type_map[types_input]))
        else:
            t.append(get_array(binfile,type_map[types_input]))
    return t
def get_nums(outputs,types_output):
    index = 0
    num_list = []
    shape = []
    #get shape

    for i in outputs:
        i.to_host()
        num = np.array(i).astype(type_map[types_output[index][0]])
        num = num.flatten()
        num.dtype = type_map[types_output[index][0]]
        shape.append(i.shape[-1])
        num_list.append(num)
        index+=1
    #to list
    i_index = 0
    nums = []
    for i in num_list:
        output_desc = len(i)
        nums.append([])
        for j in range(int(output_desc/shape[i_index])):
            for k in range(j*shape[i_index],(j+1)*shape[i_index]):
                nums[i_index].append(num_list[i_index][k])
        i_index+=1
    return (nums,shape)

def save_files(filepath,outputs,output,datatype,nums,shape,types_output):
    filepath = filepath.split('/')[-1]
    #TXT
    if datatype == 'TXT':
        i_index=0
        for i in nums:
            f = open(output+'/'+filepath.split('.')[0]+'_'+str(i_index)+".txt",'a+')
            output_desc = len(i)
            for j in range(int(output_desc/shape[i_index])):
                for k in range(j*shape[i_index],(j+1)*shape[i_index]):
                    f.write(str(nums[i_index][k])+' ')
                f.write('\n')
            i_index+=1
    else:
    #BIN
        i_index=0
        for i in outputs:
            i.to_host()
            num = np.array(i)
            #num = np.array(i).astype(type_map[types_output[i_index][0]])
            num = num.flatten()
            num.tofile(output+'/'+filepath.split('.')[0]+'_'+str(i_index)+".bin")
            i_index+=1
    

def get_array(binfile,input_type):
    files_bin = []
    #make list  files_bin
    if os.path.isdir(binfile):
        for s in os.listdir(binfile):
            files_bin.append(np.fromfile(binfile+"/"+s,dtype=input_type).flatten())
    elif os.path.isfile(binfile):
        files_bin.append(np.fromfile(binfile,dtype=input_type).flatten())
    new_files = []
    #make new bin
    for i in range(len(files_bin)):
        if i ==0:
            new_files.append(files_bin[0])
            continue
        else:
            a = files_bin[i]
            b = new_files[i-1]
            new_files.append(np.concatenate((a,b),axis=0))
    bins = np.array(new_files[-1]).astype(input_type)
    return bins

if  __name__ == '__main__':
    total_times = 0.0
    saves = True
    for i in range(loop):
        now_times = time.time()
        times = infer(saves)
        saves = False
        total_times+=times
        print("loop {0} : Inference time: {1:f} ms".format(i,times*1000))
    print("infer success!")
    print("Inference average time: {0:f} ms".format(total_times/loop*1000))


