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
import argparse
import cv2



def scan_qr(args):
    count = 0
    accuracy = 0
    qr_decoder = cv2.wechat_qrcode_WeChatQRCode()
    for root, _, files in os.walk(args.scan_dir):
        for file in files:
            path = os.path.join(root, file)
            image = cv2.imread(path, -1)
            if image is not None:
                count += 1
                retval, points = qr_decoder.detectAndDecode(image)
                if not retval:
                    print("Identify errors: {}".format(path))
                else:
                    if retval[0] == args.code_data:
                        accuracy += 1
                    else:
                        print("Identify errors: {}".format(path))

    print("Successful scans / Total scans: {} / {}".format(accuracy, count))
    print("Accuracy: {:.2f}%".format(accuracy / count * 100))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_data', 
                        help="target data of the QR codes (default: 'Thank you for reviewing our paper.')",
                        type=str,
                        default='Thank you for reviewing our paper.')
    parser.add_argument('--scan_dir', 
                        help="path to test stylized QR codes (default: './output')", 
                        type=str, 
                        default='./output')
    params = parser.parse_args()
    scan_qr(params)
