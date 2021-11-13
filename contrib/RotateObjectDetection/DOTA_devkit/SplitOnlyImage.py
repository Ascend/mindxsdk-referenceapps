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

import os
import numpy as np
import cv2
import copy
from utils import general_utils

class SplitBase():
    def __init__(self,
                 srcpath,
                 dstpath,
                 gap=200,
                 subsize=1024,
                 ext='.jpg'):
        self.srcpath = srcpath
        self.outpath = dstpath
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.srcpath = srcpath
        self.dstpath = dstpath
        self.ext = ext
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
        
    def saveimagepatches(self, img, subimgname, left, up, ext='.jpg'):
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.dstpath, subimgname + ext)
        cv2.imwrite(outdir, subimg)

    def splitsingle(self, name, rate, extent):
        img = cv2.imread(os.path.join(self.srcpath, name + extent))
        assert np.shape(img) != ()

        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '__' + str(rate) + '__'
        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]
        
        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                subimgname = outbasename + str(left) + '___' + str(up)
                self.saveimagepatches(resizeimg, subimgname, left, up)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide
        
            

    def splitdata(self, rate):     
        imagelist = general_utils.getfilefromthisrootdir(self.srcpath)
        imagenames = [general_utils.custombasename(x) for x in imagelist \
                      if (general_utils.custombasename(x) != 'Thumbs')]
        for name in imagenames:
            self.splitsingle(name, rate, self.ext)
            print(name, "split down!")

if __name__ == '__main__':
    split = SplitBase(r'/home/zhongzhi8/RotatedObjectDetection/dataSet/images',
                      r'../image')
    split.splitdata(1)