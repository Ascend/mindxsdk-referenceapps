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
import mindx.sdk as sdk
import numpy as np

image_path = "dog.jpg"                   #输入图片
device_id = 0                            #芯片ID
im = sdk.image(image_path, device_id)    #创造图片对象
resize_img = sdk.dvpp.resize(im, height=416, width=416)  #对图片进行resize处理
t = resize_img.get_tensor()   #获取图片的Tensor对象
t.to_host()
n = np.array(t)
n.tofile("dog.bin")

