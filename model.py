# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# from torch import nn
# import torch
# import math
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

import matplotlib as mpl
# we cannot use remote server's GUI, so set this
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from PIL import Image
import h5py
import numpy as np
import cv2


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot


class norm_data(nn.Module):
    def __init__(self, dim=64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim * 25)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x


class embed(nn.Module):
    def __init__(self, dim=3, dim1=128, norm=True, bias=False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x


class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias=False, ):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)

    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()

        # print(x.shape)
        # print(g.shape)
        # exit(0)
        # change = change.permute(128, 20, 20, 128).contiguous()
        # g = torch.matmul(g, change)

        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))
        return x


class cnn1x1(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x


class local(nn.Module):
    def __init__(self, dim1, dim2, bias=False):
        super(local, self).__init__()
        # self.maxpool = nn.AdaptiveMaxPool2d((1, 20))                      810
        self.se = SELayer(dim1)

        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        # 810
        self.cnn12 = nn.Conv2d(dim1, dim1, kernel_size=1, bias=bias)
        self.bn12 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        #
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)


        self.cnn3 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        # se*x
        # x1 = x1.permute(0, 3, 2, 1).contiguous()
        # x1 = se.matmul(x1)
        # x1 = x1.permute(0, 3, 2, 1).contiguous()
        #
        # x1 = self.maxpool(x1)                                                 810
        # 残差
        # res = self.cnn3(x1)
        # res = self.bn3(res)
        # 三个CNN
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.cnn12(x)
        x = self.bn12(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # attention
        x2 = self.se(x1)
        x = x2.expand_as(x) * x
        # x = res + x
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel*2, bias=False),
            nn.Softmax()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c*2, 1, 1)
        return y


# 基础卷积套餐：conv+bn+relu
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class part_pooling(nn.Module):
    def __init__(self,inchannels,partion_metrix):
        super(part_pooling,self).__init__()
        self.partion_metrix = Variable(torch.from_numpy(np.repeat(np.expand_dims(partion_metrix, 0), inchannels, axis=0)
                                                        .astype(np.float32)), requires_grad=False)
        self.joint_weight = nn.Parameter(torch.from_numpy(
            np.where(self.partion_metrix==0,np.zeros_like(self.partion_metrix),np.ones_like(self.partion_metrix))
            .astype(np.float32)), requires_grad=True)
        self.bn = nn.BatchNorm2d(inchannels)
        bn_init(self.bn, 1)
    def forward(self, x):
        n,c,t,v = x.size()
        M = self.partion_metrix.cuda(x.get_device()) * self.joint_weight
        M = M.unsqueeze(0)
        x = torch.matmul(x,M)
        x= x.view(n,c, t,-1)
        x = self.bn(x)
        return  x
# 平铺
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        # shared MLP 多层感知机
        print(gate_channels)
        self.mlp = nn.Sequential(
            Flatten(),
            # nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.Linear(128, 128 // reduction_ratio),
            nn.ReLU(),
            nn.Linear(128 // reduction_ratio, 128)
        )
        self.pool_types = pool_types

    def forward(self, x):
        # print("x",x.shape)
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                # avg_pool size: [bs, gate_channels, 1, 1]
                # print(avg_pool.shape)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                # avg_pool + max_pool
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        # print("x",x.shape)
        # print("scale",scale.shape)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    # [bs, channel, w, h] to [bs, channel, w*h]
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        # 沿着原feature的channel分别做max_pool和avg_pool，然后将两者concat
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CSAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CSAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        # print("x0",x.shape)
        # N,C,V,T=x.shape
        # print(N,C,V,T)
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        # print(x_out.shape)
        # print(x_out.shape)
        return x_out
class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,dilation=1):
        super(unit_tcn, self).__init__()
        new_ks = (kernel_size-1)*dilation+1
        pad = int((new_ks - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1),dilation=(dilation,1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1)) #在内部压缩了通道数
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
               nn.Conv2d(in_channels, out_channels, 1),
               nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)
class SGN(nn.Module):

    def __init__(self, num_classes, dataset, seg, args, bias=True):
        super(SGN, self).__init__()

        self.dim1 = 256
        self.dataset = dataset
        self.seg = seg
        num_joint = 25
        bs = args.batch_size
        if args.train:
            self.spa = self.one_hot(bs, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(bs, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()
        else:
            self.spa = self.one_hot(32 * 5, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(32 * 5, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()

        self.tem_embed = embed(self.seg, 64 * 4, norm=False, bias=bias)
        self.spa_embed = embed(num_joint, 64, norm=False, bias=bias)
        self.joint_embed = embed(6, 64, norm=True, bias=bias)
        self.dif_embed = embed(3, 64, norm=True, bias=bias)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        # self.SE1 = SELayer(self.dim1)
        # self.SE2 = SELayer(self.dim1)

        self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)

        self.compute_g1 = CSAM(self.dim1)
        self.compute_g2 = CSAM(self.dim1)
        # self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)

        self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.dim1, bias=bias)
        self.gcnr1 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)

        self.fc = nn.Linear(self.dim1 * 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)
        self.l1 = gcn_spa(6, 64, A[0], kt=1)
        self.l2 = gcn_spa(64, 64, A[0], kt=3)
        self.l3 = gcn_spa(64, 64, A[0], kt=3)

        self.l4 = gcn_spa(64, 128, A[0], stride=2, partion_m=M[0])
        self.l5 = gcn_spa(128, 128, A[1])
        self.l6 = gcn_spa(128, 128, A[1])

        self.l7 = gcn_spa(128, 256, A[1], stride=2, partion_m=M[1])
        self.l8 = gcn_spa(256, 256, A[2], dilation=5)
        self.l9 = gcn_spa(256, 256, A[2], dilation=5)

        self.drop_out = nn.Dropout(dropout)
        self.fc = nn.Linear(448, num_class)

    def forward(self, input):

        # Dynamic Representation
        bs, step, dim = input.size()
        num_joints = dim // 3
        input = input.view((bs, step, num_joints, 3))
        input = input.permute(0, 3, 2, 1).contiguous()
        dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
        dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)
        input = torch.cat([input, dif], dim=1)
        # print(input.shape)
        # exit(0)
        pos = self.joint_embed(input)
        tem1 = self.tem_embed(self.tem)
        spa1 = self.spa_embed(self.spa)
        # dif = self.dif_embed(dif)
        dy = pos
        # dy = pos + dif

        # Joint-level Module
        input = torch.cat([dy, spa1], 1)
        # print("input",input.shape)

        g1 = self.compute_g1(input)
        g2 = self.compute_g2(input)
        g1 = g1.permute(0, 3, 2, 1).contiguous()
        g2 = g2.permute(0, 3, 1, 2).contiguous()
        g = g1.matmul(g2)

        # GCN的res
        res = self.gcnr1(input, g)
        input = self.gcn1(input, g)
        input = self.gcn2(input, g)
        input = self.gcn3(input, g)
        # GCN的res
        input = input + res

        # Frame-level Module
        input = input + tem1
        # se1 = self.SE1(input)
        # se2 = self.SE1(input)
        # se1 = se1.permute(0, 3, 2, 1).contiguous()
        # se2 = se2.permute(0, 3, 1, 2).contiguous()
        # se = se1.matmul(se2)
        # input = self.cnn(input，se)
        input = self.cnn(input)

        # Classification
        output = self.maxpool(input)
        output = torch.flatten(output, 1)
        output = self.fc(output)

        return output
