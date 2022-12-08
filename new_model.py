import math

import torch
import torch.nn as nn


class OurNetwork(nn.Module):
    def __init__(self, config, act=nn.ReLU(True)):
        super(OurNetwork, self).__init__()

        self.scales = [1, 2, 3, 4]
        self.target_scale = None
        self.networks = nn.ModuleList()

        for scale in self.scales:
            self.networks.append(
                SingleNetwork(num_block=config[scale]['block'], num_feature=config[scale]['feature'], num_channel=3,
                              scale=scale, bias=True, act=act)
            )

    def set_target_scale(self, scale):
        assert scale in self.scales
        self.target_scale = scale

    def forward(self, x):
        assert self.target_scale in self.scales
        x = self.networks[self.target_scale - 1].forward(x)
        return x


class SingleNetwork(nn.Module):
    def __init__(self, num_block, num_feature, num_channel, scale, bias=True, act=nn.ReLU(True)):
        super(SingleNetwork, self).__init__()
        self.num_block = num_block
        self.num_feature = num_feature
        self.num_channel = num_channel
        self.scale = scale

        assert self.scale in [1, 2, 3, 4]

        head = []
        head.append(nn.Conv2d(in_channels=self.num_channel, out_channels=self.num_feature,
                              kernel_size=3, stride=1, padding=1, bias=bias))

        body = []
        # for _ in range(self.num_block):
        #     body.append(ResBlock(self.num_feature, bias=bias, act=act))
        for _ in range(self.num_block):
            body.append(NAFBlock(self.num_feature))
        body.append(nn.Conv2d(in_channels=self.num_feature, out_channels=self.num_feature,
                              kernel_size=3, stride=1, padding=1, bias=bias))

        tail = []
        tail.append(nn.Conv2d(in_channels=self.num_feature, out_channels=self.num_channel,
                              kernel_size=3, stride=1, padding=1, bias=bias))

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        # self.LA = Layer_Attention_Module()
        if self.scale > 1:
            self.upscale = nn.Sequential(*UpSampler(self.scale, self.num_feature, bias=bias))

    def get_output_nodes(self):
        return self.output_node

    def forward(self, x):
        # feed-forward part
        x = self.head(x)
        # body_results = list()
        # body_results.append(x)
        # for RB in self.body:
        #     x = RB(x)
        #     body_results.append(x)
        res = self.body(x)
        res += x
        # feature_LA = self.LA(torch.stack(body_results[1:-1], dim=1))
        # res = feature_LA+body_results[0]

        if self.scale > 1:
            x = self.upscale(res)
        else:
            x = res

        x = self.tail(x)
        # del body_results
        return x


class ConvReLUBlock(nn.Module):
    def __init__(self, num_feature, bias):
        super(ConvReLUBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_feature, out_channels=num_feature, kernel_size=3, stride=1, padding=1,
                              bias=bias)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class ResBlock(nn.Module):
    def __init__(self, num_feature, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, kernel_size=3):

        super(ResBlock, self).__init__()
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


class UpSampler(nn.Sequential):
    def __init__(self, scale, nFeat, bn=False, act=None, bias=True):
        super(UpSampler, self).__init__()

        modules = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                modules.append(
                    nn.Conv2d(in_channels=nFeat, out_channels=4 * nFeat, kernel_size=3, stride=1, padding=1, bias=bias))
                modules.append(nn.PixelShuffle(2))
                if bn:
                    modules.append(nn.BatchNorm2d(nFeat))
                if act:
                    modules.append(act())
        elif scale == 3:
            modules.append(
                nn.Conv2d(in_channels=nFeat, out_channels=9 * nFeat, kernel_size=3, stride=1, padding=1, bias=bias))
            modules.append(nn.PixelShuffle(3))
            if bn:
                modules.append(nn.BatchNorm2d(nFeat))
            if act:
                modules.append(act())
        else:
            raise NotImplementedError

        self.upsampler = nn.Sequential(*modules)

    def forward(self, x):
        return self.upsampler(x)

## Channel Attention (CA) Layer
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

class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""

    def __init__(self, in_dim):
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        out = self.gamma * out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x

class Layer_Attention_Module(nn.Module):
    def __init__(self):
        super(Layer_Attention_Module, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.scale = nn.Parameter(torch.zeros(1))
        self.n = 8  #config['MODEL_CONFIG']['N_RESGROUPS']
        self.c = 48
        self.conv = nn.Conv2d(self.n * self.c, self.c, kernel_size=3, padding=1)

    def forward(self, feature_group):
        b,n,c,h,w = feature_group.size()
        feature_group_reshape = feature_group.view(b, n, c * h * w)

        attention_map = torch.bmm(feature_group_reshape, feature_group_reshape.view(b, c * h * w, n))
        attention_map = self.softmax(attention_map) # N * N

        attention_feature = torch.bmm(attention_map, feature_group_reshape) # N * CHW
        b, n, chw = attention_feature.size()
        attention_feature = attention_feature.view(b,n,c,h,w)

        attention_feature = self.scale * attention_feature + feature_group
        b, n, c, h, w = attention_feature.size()
        return self.conv(attention_feature.view(b, n * c, h, w))


## NFA BLOCK
class NAFBlock(nn.Module):
    def __init__(self, c, kernel_size=3 ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=kernel_size, padding=(kernel_size // 2),
                               stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=c*2, out_channels=c, kernel_size=kernel_size, padding=(kernel_size // 2),
                               stride=1,
                               groups=1,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=kernel_size, padding=(kernel_size // 2), stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )
        self.sca = CALayer(c)
        # self.res = ResBlock(dw_channel, bias=True, act=nn.ReLU(True))
        # SimpleGate
        # self.sg = SimpleGate(2*c)

        self.conv4 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=kernel_size, padding=(kernel_size // 2),
                               stride=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=kernel_size, padding=(kernel_size // 2),
                               stride=1,
                                bias=True)
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=2*c, kernel_size=kernel_size, stride=1,
        #           padding=(kernel_size // 2), bias=True)
        # self.conv5 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=kernel_size, stride=1,
        #                        padding=(kernel_size // 2), bias=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.res(x)
        # x = self.sg(x)
        # x = self.sca(x)
        # x = x * self.sca(x)
        x = self.conv3(x)
        x = self.sca(x)
        # x = self.dropout1(x)

        # y = inp + x * self.beta
        y = inp+x
        x = self.conv4(y)
        # x = self.sg(x)
        x = self.conv5(x)
        # # x = self.dropout2(x)
        #
        # # return y + x * self.gamma
        return y+x

class SimpleGate(nn.Module):
    def __init__(self, c):
        super().__init__()
        # self.res = ResBlock(c, bias=True, act=nn.ReLU(True))
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        # x1 = self.res(x1)
        # x2 = self.res(x2)
        return x1 * x2