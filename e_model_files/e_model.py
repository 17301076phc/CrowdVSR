#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:06:03 2020

@author: Pablo Navarrete Michelini
"""
import math

import torch
import numpy as np
from torch import nn


class edgeSR_MAX(nn.Module):
    def __init__(self, model_id):
        self.model_id = model_id
        super().__init__()

        assert self.model_id.startswith('eSR-MAX_')

        parse = self.model_id.split('_')

        self.channels = int([s for s in parse if s.startswith('C')][0][1:])
        self.kernel_size = (int([s for s in parse if s.startswith('K')][0][1:]), ) * 2
        self.stride = (int([s for s in parse if s.startswith('s')][0][1:]), ) * 2

        self.pixel_shuffle = nn.PixelShuffle(self.stride[0])
        self.filter = nn.Conv2d(
            in_channels=1,
            out_channels=self.stride[0]*self.stride[1]*self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(
                (self.kernel_size[0]-1)//2,
                (self.kernel_size[1]-1)//2
            ),
            groups=1,
            bias=False,
            dilation=1
        )
        nn.init.xavier_normal_(self.filter.weight, gain=1.)
        self.filter.weight.data[:, 0, self.kernel_size[0]//2, self.kernel_size[0]//2] = 1.

    def forward(self, input):
        return self.pixel_shuffle(self.filter(input)).max(dim=1, keepdim=True)[0]


class edgeSR_TM(nn.Module):
    def __init__(self, model_id):
        self.model_id = model_id
        super().__init__()

        assert self.model_id.startswith('eSR-TM_')

        parse = self.model_id.split('_')

        self.channels = int([s for s in parse if s.startswith('C')][0][1:])
        self.kernel_size = (int([s for s in parse if s.startswith('K')][0][1:]), ) * 2
        self.stride = (int([s for s in parse if s.startswith('s')][0][1:]), ) * 2

        self.pixel_shuffle = nn.PixelShuffle(self.stride[0])
        self.softmax = nn.Softmax(dim=1)
        self.filter = nn.Conv2d(
            in_channels=1,
            out_channels=2*self.stride[0]*self.stride[1]*self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(
                (self.kernel_size[0]-1)//2,
                (self.kernel_size[1]-1)//2
            ),
            groups=1,
            bias=False,
            dilation=1
        )
        nn.init.xavier_normal_(self.filter.weight, gain=1.)
        self.filter.weight.data[:, 0, self.kernel_size[0]//2, self.kernel_size[0]//2] = 1.

    def forward(self, input):
        filtered = self.pixel_shuffle(self.filter(input))

        value, key = torch.split(filtered, [self.channels, self.channels], dim=1)
        return torch.sum(
            value * self.softmax(key),
            dim=1, keepdim=True
        )


class edgeSR_TR(nn.Module):
    def __init__(self, model_id):
        self.model_id = model_id
        super().__init__()

        assert self.model_id.startswith('eSR-TR_')
        # eSR - TR_s2_K7_C16.model
        parse = self.model_id.split('_')

        self.channels = int([s for s in parse if s.startswith('C')][0][1:])
        self.kernel_size = (int([s for s in parse if s.startswith('K')][0][1:]), ) * 2
        self.stride = (int([s for s in parse if s.startswith('s')][0][1:]), ) * 2
        # print(self.stride)
        self.pixel_shuffle = nn.PixelShuffle(self.stride[0]) # [batch,r^2c,h,w]->[batch,c,rh,rw]
        self.softmax = nn.Softmax(dim=1)
        self.filter = nn.Conv2d(
            in_channels=48, # 1
            out_channels=9*self.stride[0]*self.stride[1]*self.channels,
            # out_channels=3*self.stride[0]*self.stride[1]*self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(
                (self.kernel_size[0]-1)//2,
                (self.kernel_size[1]-1)//2
            ),
            groups=1,
            bias=False,
            dilation=1
        )
        nn.init.xavier_normal_(self.filter.weight, gain=1.)
        self.filter.weight.data[:, 0, self.kernel_size[0]//2, self.kernel_size[0]//2] = 1.

    def forward(self, input):
        filtered = self.pixel_shuffle(self.filter(input))
        sp_tube = [self.channels for _ in range(9)]
        v0,q0,k0,v1,q1,k1,v2,q2,k2 = torch.split(filtered, sp_tube, dim=1)
        x0 = torch.sum(
            v0 * self.softmax(q0 * k0),
            dim=1, keepdim=True
        )
        x1 = torch.sum(
            v1 * self.softmax(q1 * k1),
            dim=1, keepdim=True
        )
        x2 = torch.sum(
            v2 * self.softmax(q2 * k2),
            dim=1, keepdim=True
        )
        return torch.cat((x0,x1,x2),dim=1)


class edgeSR_CNN(nn.Module):
    def __init__(self, model_id):
        self.model_id = model_id
        super().__init__()

        assert self.model_id.startswith('eSR-CNN_')

        parse = self.model_id.split('_')

        self.channels = int([s for s in parse if s.startswith('C')][0][1:])
        self.stride = (int([s for s in parse if s.startswith('s')][0][1:]), ) * 2
        D = int([s for s in parse if s.startswith('D')][0][1:])
        S = int([s for s in parse if s.startswith('S')][0][1:])
        assert S>0 and D>=0

        self.softmax = nn.Softmax(dim=1)
        if D == 0:
            self.filter = nn.Sequential(
                nn.Conv2d(D, S, (3, 3), (1, 1), (1, 1)),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=S,
                    out_channels=2*self.stride[0]*self.stride[1]*self.channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    groups=1,
                    bias=False,
                    dilation=1
                ),
                nn.PixelShuffle(self.stride[0]),
            )
        else:
            self.filter = nn.Sequential(
                nn.Conv2d(1, D, (5, 5), (1, 1), (2, 2)),
                nn.Tanh(),
                nn.Conv2d(D, S, (3, 3), (1, 1), (1, 1)),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=S,
                    out_channels=2*self.stride[0]*self.stride[1]*self.channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    groups=1,
                    bias=False,
                    dilation=1
                ),
                nn.PixelShuffle(self.stride[0]),
            )

        if D == 0:
            nn.init.xavier_normal_(self.filter[0].weight, gain=1.)
            nn.init.xavier_normal_(self.filter[2].weight, gain=1.)
            self.filter[0].weight.data[:, 0, 1, 1] = 1.
            self.filter[2].weight.data[:, 0, 1, 1] = 1.
        else:
            nn.init.xavier_normal_(self.filter[0].weight, gain=1.)
            nn.init.xavier_normal_(self.filter[2].weight, gain=1.)
            nn.init.xavier_normal_(self.filter[4].weight, gain=1.)
            self.filter[0].weight.data[:, 0, 2, 2] = 1.
            self.filter[2].weight.data[:, 0, 1, 1] = 1.
            self.filter[4].weight.data[:, 0, 1, 1] = 1.

    def forward(self, input):
        filtered = self.filter(input)

        value, key = torch.split(filtered, [self.channels, self.channels], dim=1)
        return torch.sum(
            value * self.softmax(key*key),
            dim=1, keepdim=True
        )


class ENetwork(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.networks = nn.ModuleList()
        self.model_id=model_id
        parse = self.model_id.split('_')

        self.scale = (int([s for s in parse if s.startswith('s')][0][1:]),) * 2
        self.c = 48
        self.layer=edgeSR_TR(model_id=model_id)
        self.rca = RCABlock(self.c)

        # upscale=[]
        # scale = self.scale[0]
        # nFeat=3
        # if (scale & (scale - 1)) == 0:
        #     for _ in range(int(math.log(scale, 2))):
        #         upscale.append(
        #             nn.Conv2d(in_channels=nFeat, out_channels=4 * nFeat, kernel_size=3, stride=1, padding=1))
        #         upscale.append(nn.PixelShuffle(2))
        # elif scale == 3:
        #     upscale.append(
        #         nn.Conv2d(in_channels=nFeat, out_channels=9 * nFeat, kernel_size=3, stride=1, padding=1))
        #     upscale.append(nn.PixelShuffle(3))
        # self.headupscale = nn.Sequential(*upscale)

        head = []
        head.append(nn.Conv2d(in_channels=3, out_channels=self.c,
                              kernel_size=3, stride=1, padding=1))
        head.append(nn.Conv2d(in_channels=self.c, out_channels=self.c,
                              kernel_size=3, stride=1, padding=1))

        body = []
        for _ in range(8):
            body.append(self.rca)
        body.append(nn.Conv2d(in_channels=self.c, out_channels=self.c,
                              kernel_size=3, stride=1, padding=1))

        tail = []
        tail.append(self.layer)

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res+=x
        x = self.tail(res)
        return x


class ClassicUpscale(nn.ConvTranspose2d):
    def __init__(self, channels, stride, trainable=True, pixel_shuffle=False):
        assert len(stride) == 2
        assert isinstance(stride[0], int) and isinstance(stride[1], int)
        self.channels = channels
        self.stride = stride
        self.trainable = trainable
        self.pshuffle = pixel_shuffle

        shift = (-1.5, -1.5)
        fh = np.asarray([
            kernel_cubic(self.stride[1], shift[1])
        ]) / self.stride[1]
        fv = np.asarray([
            kernel_cubic(self.stride[0], shift[0])
        ]) / self.stride[0]

        f2d = fh * fv.T

        if self.pshuffle:
            pad_extra = [
                self.stride[0] - f2d.shape[0] % self.stride[0],
                self.stride[1] - f2d.shape[1] % self.stride[1]
            ]
            fh = np.asarray([
                np.concatenate((kernel_cubic(self.stride[1], shift[1]), np.zeros(pad_extra[1])))
            ]) / self.stride[1]
            fv = np.asarray([
                np.concatenate((kernel_cubic(self.stride[0], shift[0]), np.zeros(pad_extra[0])))
            ]) / self.stride[0]
            f2d = fh * fv.T

            super().__init__(
                in_channels=self.channels,
                out_channels=self.stride[0]*self.stride[1]*self.channels,
                kernel_size=(f2d.shape[0]//stride[0], f2d.shape[1]//stride[1]),
                stride=1,
                padding=((f2d.shape[0]//stride[0])//2, (f2d.shape[1]//stride[1])//2),
                output_padding=0,
                groups=self.channels,
                bias=False,
                dilation=1
            )
            self.weight.data.fill_(0)
            pus = nn.PixelUnshuffle(stride[0])
            for c in range(self.channels):
                self.weight.data[c, :, :, :] = stride[0] * stride[1] * pus(torch.FloatTensor(f2d).unsqueeze(0).unsqueeze(0))[0, :]
            self.pixel_shuffle = nn.PixelShuffle(stride[0])
            assert stride[0] == stride[1]
        else:
            super().__init__(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=(f2d.shape[0]+1, f2d.shape[1]+1),
                stride=stride,
                padding=(f2d.shape[0]-stride[0])//2+1,
                output_padding=((self.stride[0]+1)%2, (self.stride[1]+1)%2),
                groups=self.channels,
                bias=False,
                dilation=1
            )
            self.weight.data.fill_(0)
            for c in range(self.channels):
                self.weight.data[c, 0, :-1, :-1] = stride[0] * stride[1] * torch.FloatTensor(f2d)

        if not self.trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.pshuffle:
            return self.pixel_shuffle(super().forward(x))
        return super().forward(x)

    def __repr__(self):
        s = '{name}({channels}, stride={stride},' \
            ' trainable={trainable}, kernel_size=%dx%dx%dx%d)' % \
            (self.weight.shape[0], self.weight.shape[1],
             self.weight.shape[2], self.weight.shape[3])
        return s.format(name=self.__class__.__name__, **self.__dict__)

class RCABlock(nn.Module):
    def __init__(self, num_feature, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, kernel_size=3):

        super(RCABlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(
                nn.Conv2d(in_channels=num_feature, out_channels=num_feature, kernel_size=kernel_size, stride=1,
                          padding=(kernel_size // 2), bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(num_feature))
            if i == 0:
                modules_body.append(act)
        self.ca = CALayer(num_feature)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        if self.res_scale != 1:
            res = self.body(x).mul(self.res_scale)
        else:
            res = self.body(x)
        res = self.ca(res)
        res += x

        return res

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def kernel_cubic(zoom, phase, length=None):
    assert zoom > 0
    lower_bound = np.ceil(-2*zoom-phase)
    higher_bound = np.floor(2*zoom-phase)
    anchor = max(abs(lower_bound), abs(higher_bound))
    index = np.arange(-anchor+1, anchor+1)
    if length is not None:
        assert length >= 2*anchor
        anchor = np.ceil(length/2)
        index = np.arange(-anchor+1, length-anchor+1)
    pos = abs(index+phase) / zoom
    kernel = np.zeros(np.size(pos))
    idx = (pos < 2)
    kernel[idx] = -0.5 * pos[idx]**3 + 2.5 * pos[idx]**2 - 4*pos[idx] + 2
    idx = (pos < 1)
    kernel[idx] = 1.5 * pos[idx]**3 - 2.5 * pos[idx]**2 + 1
    kernel = kernel * zoom / np.sum(kernel)
    return kernel


class Classic(nn.Module):
    def __init__(self, model_id):
        self.model_id = model_id
        super().__init__()

        assert self.model_id.startswith('Bicubic')

        parse = self.model_id.split('_')

        self.stride = (int([s for s in parse if s.startswith('s')][0][1:]), ) * 2
        self.pshuffle = ('PS' in parse)

        if self.pshuffle:
            self.net = ClassicUpscale(
                1, self.stride, trainable=True, pixel_shuffle=True
            )
        else:
            self.net = ClassicUpscale(
                1, self.stride, trainable=True, pixel_shuffle=False
            )

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def forward(self, input):
        return self.net(input)