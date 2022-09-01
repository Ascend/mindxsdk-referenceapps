# Copyright (c) 2022. Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import math
from collections import OrderedDict
import torch
import torch.onnx
from torch import nn
import torch.nn.functional as F

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, \
        base_width=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            base_width: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3,
                         stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes *
                               self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, xing):
        residual = xing

        out = self.conv1(xing)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                s_p = spx[i]
            else:
                s_p = s_p + spx[i]
            s_p = self.convs[i](s_p)
            s_p = self.relu(self.bns[i](s_p))
            if i == 0:
                out = s_p
            else:
                out = torch.cat((out, s_p), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(xing)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, base_width=26, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.base_width = base_width
        self.scale = 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.f_c = nn.Linear(512 * block.expansion, num_classes)

        for ming in self.modules():
            if isinstance(ming, nn.Conv2d):
                nn.init.kaiming_normal_(
                    ming.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(ming, nn.BatchNorm2d):
                nn.init.constant_(ming.weight, 1)
                nn.init.constant_(ming.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', base_width=self.base_width, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                          base_width=self.base_width, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, xing):
        xing = self.conv1(xing)
        xing = self.bn1(xing)
        xing = self.relu(xing)
        xing = self.maxpool(xing)

        xing = self.layer1(xing)
        xing = self.layer2(xing)
        xing = self.layer3(xing)
        xing = self.layer4(xing)

        xing = self.avgpool(xing)
        xing = xing.view(xing.size(0), -1)
        xing = self.f_c(xing)

        return xing


def res2net50_v1b(**kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], base_width=26, **kwargs)
    return model


def res2net101_v1b(**kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 23, 3],
                    base_width=26, **kwargs)
    return model


def res2net50_v1b_26w_4s(**kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], base_width=26, **kwargs)
    return model


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xing):
        xing = self.conv(xing)
        xing = self.batch_norm(xing)
        return xing


class RFBModified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFBModified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel,
                        kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel,
                        kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel,
                        kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel,
                        kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel,
                        kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel,
                        kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, xing):
        xing0 = self.branch0(xing)
        xing1 = self.branch1(xing)
        xing2 = self.branch2(xing)
        xing3 = self.branch3(xing)
        x_cat = self.conv_cat(torch.cat((xing0, xing1, xing2, xing3), 1))

        xing = self.relu(x_cat + self.conv_res(xing))
        return xing


class Aggregation(nn.Module):
    def __init__(self, channel):
        super(Aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, xing1, xing2, xing3):
        x1_1 = xing1
        x2_1 = self.conv_upsample1(self.upsample(xing1)) * xing2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(xing1))) \
            * self.conv_upsample3(self.upsample(xing2)) * xing3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        xing = self.conv4(x3_2)
        xing = self.conv5(xing)

        return xing


class PraNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32):
        super(PraNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=False)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFBModified(512, channel)
        self.rfb3_1 = RFBModified(1024, channel)
        self.rfb4_1 = RFBModified(2048, channel)
        # ---- Partial Decoder ----
        self.agg1 = Aggregation(channel)
        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, xing):
        xing = self.resnet.conv1(xing)
        xing = self.resnet.bn1(xing)
        xing = self.resnet.relu(xing)
        xing = self.resnet.maxpool(xing)      # bs, 64, 88, 88
        # ---- low-level features ----
        xing1 = self.resnet.layer1(xing)      # bs, 256, 88, 88
        xing2 = self.resnet.layer2(xing1)     # bs, 512, 44, 44

        xing3 = self.resnet.layer3(xing2)     # bs, 1024, 22, 22
        xing4 = self.resnet.layer4(xing3)     # bs, 2048, 11, 11
        x2_rfb = self.rfb2_1(xing2)        # channel -> 32
        x3_rfb = self.rfb3_1(xing3)        # channel -> 32
        x4_rfb = self.rfb4_1(xing4)        # channel -> 32

        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        ra5_feat = ra5_feat.reshape(ra5_feat.size(1), ra5_feat.size(
            0), ra5_feat.size(2), ra5_feat.size(3))

        lateral_map_5 = F.interpolate(
            ra5_feat, scale_factor=8, mode='bilinear')
        lateral_map_5 = lateral_map_5.reshape(lateral_map_5.size(
            1), lateral_map_5.size(0), lateral_map_5.size(2), lateral_map_5.size(3))

        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear')
        crop_4 = crop_4.reshape(crop_4.size(1), crop_4.size(
            0), crop_4.size(2), crop_4.size(3))
        xing = -1*(torch.sigmoid(crop_4)) + 1
        xing = xing.expand(-1, 2048, -1, -1).mul(xing4)
        xing = self.ra4_conv1(xing)
        xing = F.relu(self.ra4_conv2(xing))
        xing = F.relu(self.ra4_conv3(xing))
        xing = F.relu(self.ra4_conv4(xing))
        ra4_feat = self.ra4_conv5(xing)
        xing = ra4_feat + crop_4
        temp = xing.reshape(xing.size(1), xing.size(0), xing.size(2), xing.size(3))
        # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)
        lateral_map_4 = F.interpolate(temp, scale_factor=32, mode='bilinear')
        lateral_map_4 = lateral_map_4.reshape(lateral_map_4.size(
            1), lateral_map_4.size(0), lateral_map_4.size(2), lateral_map_4.size(3))

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(temp, scale_factor=2, mode='bilinear')
        crop_3 = crop_3.reshape(crop_3.size(1), crop_3.size(
            0), crop_3.size(2), crop_3.size(3))
        xing = -1*(torch.sigmoid(crop_3)) + 1
        xing = xing.expand(-1, 1024, -1, -1).mul(xing3)
        xing = self.ra3_conv1(xing)
        xing = F.relu(self.ra3_conv2(xing))
        xing = F.relu(self.ra3_conv3(xing))
        ra3_feat = self.ra3_conv4(xing)
        xing = ra3_feat + crop_3
        temp1 = xing.reshape(xing.size(1), xing.size(0), xing.size(2), xing.size(3))
        # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)
        lateral_map_3 = F.interpolate(temp1, scale_factor=16, mode='bilinear')
        lateral_map_3 = lateral_map_3.reshape(lateral_map_3.size(
            1), lateral_map_3.size(0), lateral_map_3.size(2), lateral_map_3.size(3))

        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(temp1, scale_factor=2, mode='bilinear')
        crop_2 = crop_2.reshape(crop_2.size(1), crop_2.size(
            0), crop_2.size(2), crop_2.size(3))
        xing = -1*(torch.sigmoid(crop_2)) + 1
        xing = xing.expand(-1, 512, -1, -1).mul(xing2)
        xing = self.ra2_conv1(xing)
        xing = F.relu(self.ra2_conv2(xing))
        xing = F.relu(self.ra2_conv3(xing))
        ra2_feat = self.ra2_conv4(xing)
        xing = ra2_feat + crop_2
        temp2 = xing.reshape(xing.size(1), xing.size(0), xing.size(2), xing.size(3))
        # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        lateral_map_2 = F.interpolate(temp2, scale_factor=8, mode='bilinear')
        lateral_map_2 = lateral_map_2.reshape(lateral_map_2.size(
            1), lateral_map_2.size(0), lateral_map_2.size(2), lateral_map_2.size(3))

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2


def proc_node_module(checkpoint, attr_name):
    new_state_dict = OrderedDict()
    for k, ving in checkpoint[attr_name].items():
        if k[0:7] == "module.":
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = ving
    return new_state_dict


def convert(pth_file_path, onnx_file_path):
    model = PraNet()
    pretrained_dict = torch.load(pth_file_path, map_location="cpu")
    model.load_state_dict(
        {k.replace('module.', ''): ving for k, ving in pretrained_dict.items()})
    if "f_c.weight" in pretrained_dict:
        pretrained_dict.pop('f_c.weight')
        pretrained_dict.pop('f_c.bias')
    model.load_state_dict(pretrained_dict)
    model.eval()

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 352, 352)
    torch.onnx.export(model, dummy_input, onnx_file_path,
                      input_names=input_names, dynamic_axes=dynamic_axes, output_names=output_names,
                      opset_version=11)


if __name__ == "__main__":
    PTHPATH = sys.argv[1]
    ONNXPATH = sys.argv[2]
    convert(PTHPATH, ONNXPATH)
