# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
from torch.nn import functional as F
import math

# from Encoder import Encoder


import pdb

class Singe_prototype(nn.Module):


    def __init__(self, in_c, num_p):
        super(Singe_prototype, self).__init__()
        self.num_cluster = num_p
        self.netup = torch.nn.Sequential(
                torch.nn.Conv2d(in_c, num_p, 3, padding=1)
                )
        self.centroids = torch.nn.Parameter(torch.rand(num_p, in_c))   #生成（24, 256)可训练学习的张量

        self.upfc = torch.nn.Linear(num_p*in_c, in_c)
        self.eca = eca_block(256)
        self.transform = torch.nn.Sequential(
            nn.Conv2d(2*in_c, in_c, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_c, in_c, kernel_size=1),
            nn.ReLU(inplace=False),
            )

    def UP(self, scene):
        x = scene

        N, C, W, H = x.shape[0:]

        x = F.normalize(x, p=2, dim=1)          #对x做正则化，除2范数
        soft_assign = self.netup(x)                #通道数变为24

        soft_assign = F.softmax(soft_assign, dim=1)  #通道注意力机制
        soft_assign = soft_assign.view(soft_assign.shape[0], soft_assign.shape[1], -1)
        #调整图的大小
        x_flatten = x.view(N, C, -1)

        centroid = self.centroids       #生成（24, 256)可训练学习的张量

        x1 = x_flatten.expand(self.num_cluster, -1, -1, -1).permute(1, 0, 2, 3) #对维度进行扩展
        x2 = centroid.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)#在0处增加一个维度

        residual = x1 - x2
        residual = residual * soft_assign.unsqueeze(2)
        up = residual.sum(dim=-1)

        up = F.normalize(up, p=2, dim=2)
        up = up.view(x.size(0), -1)
        up = F.normalize(up, p=2, dim=1)

        up = self.upfc(up).unsqueeze(2).unsqueeze(3).repeat(1,1,W,H)

        return up, centroid

    def forward(self, feature):
        feature = self.eca(feature)
        up, centroid = self.UP(feature)
        up = self.eca(up)
        new_feature = torch.cat((feature, up), dim=1)
        new_feature = self.transform(new_feature)

        return new_feature

class eca_block(nn.Module):

    def __init__(self, in_channels, gamma=2, b=1):
        super(eca_block, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        output = self.fgp(x)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(output)
        output = torch.multiply(x, output)
        return output


# 定义了一个名为UP的方法，该方法接受输入的特征图并返回处理后的特征图和聚类中心。方法的主要步骤包括：
#
# a. 将输入特征图进行2范数归一化。
#
# b. 通过一个卷积层（self.netup）计算特征图中每个像素点对应的K个原型的分数，即软分配概率。
#
# c. 将软分配概率进行softmax归一化，并将其转换为与输入特征图具有相同空间维度的张量。
#
# d. 将输入特征图展开为(N, C, H * W)的形状。
#
# e. 将聚类中心张量扩展为(N, num_p, C)的形状。
#
# f. 计算每个原型与特征图像素之间的残差，并将其乘以软分配概率。
#
# g. 对残差进行求和，得到每个原型的加权残差，形状为(N, num_p, C)。
#
# h. 对每个加权残差进行2范数归一化，并在最后的维度上重新调整形状为(N, num_p * C)。
#
# i. 将加权残差传递给全连接层，并将其输出形状调整为(N, num_p, C)。
#
# j. 将全连接层的输出张量扩展为与输入特征图具有相同的空间维度，并返回该张量和聚类中心。