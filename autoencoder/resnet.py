from __future__ import absolute_import, division, print_function, unicode_literals

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 3D Resnet implementation inspired by https://github.com/pykao/Modified-3D-UNet-Pytorch/blob/master/model.py

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

class GroupNorm3D(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5 ):
        super(GroupNorm3D, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        # magic number
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W, L = x.size()
        G = self.num_groups
        # make it work also for shallow nets
        G = min(G, C)
        assert C % G == 0

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W,L)
        return x * self.weight + self.bias

class BasicBlockEncoder(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groupNorm=True):
        super(BasicBlockEncoder, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        if not groupNorm:
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            self.bn1 = GroupNorm3D(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        if not groupNorm:
            self.bn2 = nn.BatchNorm3d(planes)
        else:
            self.bn2 = GroupNorm3D(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlockDecoder(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, groupNorm=True):
        super(BasicBlockDecoder, self).__init__()

        self.conv1 = conv3x3x3(inplanes, inplanes)
        if not groupNorm:
            self.bn1 = nn.BatchNorm3d(inplanes)
        else:
            self.bn1 = GroupNorm3D(inplanes)
        self.relu = nn.ReLU(inplace=True)

        if stride > 1:
            self.conv2 = nn.ConvTranspose3d(inplanes, planes , kernel_size=2, stride=stride)
        else:
            self.conv2 = conv3x3x3(inplanes, planes, stride)
        if not groupNorm:
            self.bn2 = nn.BatchNorm3d(planes)
        else:
            self.bn2 = GroupNorm3D(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

class Encoder(nn.Module):

    def __init__(self,
                i_planes,
                input_size,
                layers,
                base_out_total,
                block = BasicBlockEncoder,
                start_planes=8,
                start_padding = 0,
                group_norm = True):

        super(Encoder, self).__init__()
        self.base_out_total = base_out_total
        self.group_norm = group_norm
        self.inplanes = start_planes
        self.conv1 = nn.Conv3d(i_planes, start_planes, kernel_size=3, stride = 1, padding = 1, bias=False)
        if not self.group_norm:
            self.bn1 = nn.BatchNorm3d(start_planes)
        else:
            self.bn1 = GroupNorm3D(start_planes)
        self.relu = nn.ReLU(inplace=True)


        self.layers =[]
        self.layers.append(self._make_layer(block, start_planes, layers[0]))
        for i in range(1,len(layers)):
            self.layers.append(self._make_layer(block, start_planes*(2**i), layers[i], stride=2))
        self.layers = nn.Sequential(*self.layers)

        self.conv2 = nn.Conv3d(start_planes *(2**(len(layers)-1)), start_planes *(2**(len(layers)-1)), kernel_size=1, padding=0)
        if not self.group_norm:
            self.bn2 = nn.BatchNorm2d(start_planes *(2**(len(layers)-1)))
        else:
            self.bn2 = GroupNorm3D(start_planes *(2**(len(layers)-1)))


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            if not self.group_norm:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), GroupNorm3D(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layers(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x.view([-1,self.base_out_total])

        return x


class EncoderTiny(nn.Module):

    def __init__(self,
                layers,
                base_out_total,
                block = BasicBlockEncoder,
                start_planes=8,
                start_padding = 0,
                group_norm = True):

        super(EncoderTiny, self).__init__()
        self.base_out_total = base_out_total
        self.group_norm = group_norm
        self.inplanes = start_planes

        self.layer1 = self._make_layer(block, start_planes, layers[0])
        self.layer2 = self._make_layer(block, start_planes*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, start_planes*4, layers[2], stride=2)

        self.conv2 = nn.Conv3d(start_planes * 4, start_planes * 4, kernel_size=1, padding=0)
        if not self.group_norm:
            self.bn2 = nn.BatchNorm2d(start_planes * 4)
        else:
            self.bn2 = GroupNorm3D(start_planes * 4)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            if not self.group_norm:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), GroupNorm3D(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x.view([-1,self.base_out_total])

        return x

class Decoder(nn.Module):

    def __init__(self,
                o_planes,
                shape,
                layers,
                block = BasicBlockDecoder,
                group_norm = True):

        super(Decoder, self).__init__()
        self.group_norm = group_norm
        self.shape = shape

        self.conv1 = nn.Conv3d(shape[0], shape[0]*2, kernel_size=3, stride = 1, padding = 1, bias=False)
        if not self.group_norm:
            self.bn1 = nn.BatchNorm3d(shape[0]*2)
        else:
            self.bn1 = GroupNorm3D(shape[0]*2)
        self.relu = nn.ReLU(inplace=True)

        self.inplanes = shape[0]*2
        self.layers = []
        for i in range(len(layers)-1):
            self.layers.append(self._make_layer(block, shape[0]*2 // (2**(i+1)), layers[len(layers) -1 -i], stride=2))
        self.layers.append (self._make_layer(block, shape[0]*2 // (2**(len(layers)-1)), layers[0]))
        self.layers = nn.Sequential(*self.layers)

        self.conv2 = nn.Conv3d(shape[0]*2 // (2**(len(layers)-1)), o_planes, kernel_size=3, stride = 1, padding = 1, bias=False)



    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if not self.group_norm:
                upsample = nn.Sequential(
                    nn.ConvTranspose3d(self.inplanes, planes * block.expansion , kernel_size=2, stride=stride, bias = False),
                    nn.BatchNorm3d(planes * block.expansion))
            else:
                upsample = nn.Sequential(
                    nn.ConvTranspose3d(self.inplanes, planes * block.expansion , kernel_size=2, stride=stride, bias = False),
                     GroupNorm3D(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view([-1,self.shape[0],self.shape[1],self.shape[2],self.shape[3]])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layers(x)
        x = self.conv2(x)

        return x

class DecoderTiny(nn.Module):

    def __init__(self,
                o_planes,
                shape,
                layers,
                block = BasicBlockDecoder,
                group_norm = True):

        super(DecoderTiny, self).__init__()
        self.group_norm = group_norm
        self.inplanes = shape[0]
        self.shape = shape

        self.layer1 = self._make_layer(block, shape[0], layers[3], stride=2)
        self.layer2 = self._make_layer(block, shape[0] // 2, layers[2], stride=2)
        self.layer3 = self._make_layer(block, shape[0] // 4, layers[1], stride=2)



    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if not self.group_norm:
                upsample = nn.Sequential(
                    nn.ConvTranspose3d(self.inplanes, planes * block.expansion , kernel_size=2, stride=stride, bias = False),
                    nn.BatchNorm3d(planes * block.expansion))
            else:
                upsample = nn.Sequential(
                    nn.ConvTranspose3d(self.inplanes, planes * block.expansion , kernel_size=2, stride=stride, bias = False),
                     GroupNorm3D(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view([-1,self.shape[0],self.shape[1],self.shape[2],self.shape[3]])
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
