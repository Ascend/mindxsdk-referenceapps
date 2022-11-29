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
import argparse
from artcoder import artcoder
import utils as utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--style_img_path', 
                        help="path to input style target (default: '../datasets/style/texture1.1.jpg')", 
                        type=str, 
                        default='../datasets/style/texture1.1.jpg')
    parser.add_argument('--content_img_path', 
                        help="path to input content target (default: '../datasets/content/boy.jpg')", 
                        type=str, 
                        default='../datasets/content/boy.jpg')
    parser.add_argument('--code_img_path', 
                        help="path to input code target (default: '../datasets/code/boy.jpg')", 
                        type=str, 
                        default='../datasets/code/boy.jpg')
    parser.add_argument('--output_dir', help='path to save output stylized QR code', 
                        type=str,
                        default='./output/')
    parser.add_argument('--learning_rate',
                        help='learning rate (default: 0.01)',
                        type=int, 
                        default=0.05)
    parser.add_argument('--style_weight', 
                        help='style_weight', 
                        type=int, 
                        default=1e6)
    parser.add_argument('--content_weight', 
                        help='content_weight', 
                        type=int, 
                        default=1e7)
    parser.add_argument('--code_weight', 
                        help='code_weight', 
                        type=int, 
                        default=1e12)
    parser.add_argument('--module_size',
                        help='the resolution of each square module of a QR code (default: 16)',
                        type=int, 
                        default=16)
    parser.add_argument('--module_number',
                        help='Number of QR code modules per side (default: 37)',
                        type=int, 
                        default=37)
    parser.add_argument('--epoch', 
                        help='epoch number (default: 20000)', 
                        type=int,
                        default=100)
    parser.add_argument('--discriminate_b',
                        help="for black modules, pixels' gray values under discriminate_b will be discriminated \
                            to error modules to activate sub-code-losses (discriminate_b in [0-128])",
                        type=int,
                        default=70)
    parser.add_argument('--discriminate_w',
                        help="for white modules, pixels' gray values over discriminate_w will be discriminated \
                            to error modules to activate sub-code-losses (discriminate_w in [128-255])",
                        type=int,
                        default=180)
    parser.add_argument('--correct_b',
                        help="for black module, correct error modules' gray value to correct_b \
                            (correct_b < discriminate_b)",
                        type=int,
                        default=40)
    parser.add_argument('--correct_w',
                        help="for white module, correct error modules' gray value to correct_w \
                            (correct_w > discriminate_w)",
                        type=int,
                        default=220)
    parser.add_argument('--use_activation_mechanism',
                        help="whether to use the activation mechanism \
                            (1 means use and other numbers mean not)",
                        type=int,
                        default=1)

    args = parser.parse_args()
    utils.print_options(opt=args)

    artcoder(style_img_path=args.style_img_path, content_img_path=args.content_img_path, code_path=args.code_img_path,
             output_dir=args.output_dir, learning_rate=args.learning_rate, content_weight=args.content_weight,
             style_weight=args.style_weight, code_weight=args.code_weight, module_size=args.module_size,
             module_num=args.module_number, epochs=args.epoch, dis_b=args.discriminate_b, dis_w=args.discriminate_w,
             correct_b=args.correct_b, correct_w=args.correct_w, use_activation_mechanism=args.use_activation_mechanism)
