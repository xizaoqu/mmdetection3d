# Copyright (c) OpenMMLab. All rights reserved.
r"""Modified from Cylinder3D.

Please refer to `Cylinder3D github page <http://www.semantic-
kitti.org/dataset.html>`_ for details
"""

import numpy as np
import torch
from mmcv.cnn import build_norm_layer
from mmcv.ops import (SparseConv3d, SparseConvTensor, SparseInverseConv3d,
                      SubMConv3d)
from mmengine.model import BaseModule
from torch import nn as nn

from mmdet3d.registry import MODELS


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=stride,
        padding=(0, 1, 1),
        bias=False,
        indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 1, 3),
        stride=stride,
        padding=(0, 0, 1),
        bias=False,
        indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 1),
        stride=stride,
        padding=(0, 1, 0),
        bias=False,
        indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=(3, 1, 1),
        stride=stride,
        padding=(1, 0, 0),
        bias=False,
        indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=(3, 1, 3),
        stride=stride,
        padding=(1, 0, 1),
        bias=False,
        indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=False,
        indice_key=indice_key)


class AsymmResBlock(nn.Module):
    """Asymmetrical Residual Block.

    Args:
        in_channels (int): Input channels of the block.
        out_channels (int): Output channels of the block.
        norm_cfg (dict): Dictionary to construct and config
            norm layer.
        indice_key (str, optional): Name of indice tables. Default: None.
    """

    def __init__(self, in_channels, out_channels, norm_cfg, indice_key=None):
        super().__init__()

        self.conv0_0 = conv1x3(
            in_channels, out_channels, indice_key=indice_key + 'bef')
        self.bn0_0 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act0_0 = nn.LeakyReLU()

        self.conv0_1 = conv3x1(
            out_channels, out_channels, indice_key=indice_key + 'bef')
        self.bn0_1 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act0_1 = nn.LeakyReLU()

        self.conv1_0 = conv3x1(
            in_channels, out_channels, indice_key=indice_key + 'bef')
        self.act1_0 = nn.LeakyReLU()
        self.bn1_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_1 = conv1x3(
            out_channels, out_channels, indice_key=indice_key + 'bef')
        self.act1_1 = nn.LeakyReLU()
        self.bn1_1 = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x):
        """Forward pass."""
        shortcut = self.conv0_0(x)

        shortcut.features = self.act0_0(shortcut.features)
        shortcut.features = self.bn0_0(shortcut.features)

        shortcut = self.conv0_1(shortcut)
        shortcut.features = self.act0_1(shortcut.features)
        shortcut.features = self.bn0_1(shortcut.features)

        res = self.conv1_0(x)
        res.features = self.act1_0(res.features)
        res.features = self.bn1_0(res.features)

        res = self.conv1_1(res)
        res.features = self.act1_1(res.features)
        res.features = self.bn1_1(res.features)

        res.features = res.features + shortcut.features

        return res


class AsymmeDownBlock(nn.Module):
    """Asymmetrical DownSample Block.

    Args:
       in_channels (int): Input channels of the block.
       out_channels (int): Output channels of the block.
       norm_cfg (dict): Dictionary to construct and config
           norm layer.
       pooling (bool, optional): Whether pooling features at the end of
           block. Defaults: True.
       height_pooling (bool, optional): Whether pooling features at
           the height dimension. Defaults: False.
       indice_key (str, optional): Name of indice tables. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg,
                 pooling=True,
                 height_pooling=False,
                 indice_key=None):
        super().__init__()
        self.pooling = pooling

        self.conv0_0 = conv3x1(
            in_channels, out_channels, indice_key=indice_key + 'bef')
        self.act0_0 = nn.LeakyReLU()
        self.bn0_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv0_1 = conv1x3(
            out_channels, out_channels, indice_key=indice_key + 'bef')
        self.act0_1 = nn.LeakyReLU()
        self.bn0_1 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_0 = conv1x3(
            in_channels, out_channels, indice_key=indice_key + 'bef')
        self.act1_0 = nn.LeakyReLU()
        self.bn1_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_1 = conv3x1(
            out_channels, out_channels, indice_key=indice_key + 'bef')
        self.act1_1 = nn.LeakyReLU()
        self.bn1_1 = build_norm_layer(norm_cfg, out_channels)[1]

        if pooling:
            if height_pooling:
                self.pool = SparseConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    indice_key=indice_key,
                    bias=False)
            else:
                self.pool = SparseConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=(2, 2, 1),
                    padding=1,
                    indice_key=indice_key,
                    bias=False)

    def forward(self, x):
        """Forward pass."""
        shortcut = self.conv0_0(x)
        shortcut.features = self.act0_0(shortcut.features)
        shortcut.features = self.bn0_0(shortcut.features)

        shortcut = self.conv0_1(shortcut)
        shortcut.features = self.act0_1(shortcut.features)
        shortcut.features = self.bn0_1(shortcut.features)

        res = self.conv1_0(x)
        res.features = self.act1_0(res.features)
        res.features = self.bn1_0(res.features)

        res = self.conv1_1(res)
        res.features = self.act1_1(res.features)
        res.features = self.bn1_1(res.features)

        res.features = res.features + shortcut.features

        if self.pooling:
            pooled_res = self.pool(res)
            return pooled_res, res
        else:
            return res


class AsymmeUpBlock(nn.Module):
    """Asymmetrical UpSample Block.

    Args:
       in_channels (int): Input channels of the block.
       out_channels (int): Output channels of the block.
       norm_cfg (dict): Dictionary to construct and config
           norm layer.
       indice_key (str, optional): Name of indice tables. Default: None.
       up_key (str, optional): Name of indice tables used in
           SparseInverseConv3d. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg,
                 indice_key=None,
                 up_key=None):
        super().__init__()

        self.trans_conv = conv3x3(
            in_channels, out_channels, indice_key=indice_key + 'new_up')
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1 = conv1x3(out_channels, out_channels, indice_key=indice_key)
        self.act1 = nn.LeakyReLU()
        self.bn1 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv2 = conv3x1(out_channels, out_channels, indice_key=indice_key)
        self.act2 = nn.LeakyReLU()
        self.bn2 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv3 = conv3x3(out_channels, out_channels, indice_key=indice_key)
        self.act3 = nn.LeakyReLU()
        self.bn3 = build_norm_layer(norm_cfg, out_channels)[1]

        self.up_subm = SparseInverseConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            indice_key=up_key,
            bias=False)

    def forward(self, x, skip):
        """Forward pass."""
        x_trans = self.trans_conv(x)
        x_trans.features = self.trans_act(x_trans.features)
        x_trans.features = self.trans_bn(x_trans.features)

        # upsample
        up = self.up_subm(x_trans)

        up.features = up.features + skip.features

        up = self.conv1(up)
        up.features = self.act1(up.features)
        up.features = self.bn1(up.features)

        up = self.conv2(up)
        up.features = self.act2(up.features)
        up.features = self.bn2(up.features)

        up = self.conv3(up)
        up.features = self.act3(up.features)
        up.features = self.bn3(up.features)

        return up


class DDCMBlock(nn.Module):
    """Dimension-Decomposition based Context Modeling.

    Args:
        in_channels (int): Input channels of the block.
        out_channels (int): Output channels of the block.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        indice_key (str, optional): Name of indice tables. Default: None.
    """

    def __init__(self, in_channels, out_channels, norm_cfg, indice_key=None):
        super().__init__()

        self.conv1 = conv3x1x1(
            in_channels, out_channels, indice_key=indice_key + 'bef')
        self.bn1 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act1 = nn.Sigmoid()

        self.conv2 = conv1x3x1(
            in_channels, out_channels, indice_key=indice_key + 'bef')
        self.bn2 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act2 = nn.Sigmoid()

        self.conv3 = conv1x1x3(
            in_channels, out_channels, indice_key=indice_key + 'bef')
        self.bn3 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        """Forward pass."""
        shortcut = self.conv1(x)
        shortcut.features = self.bn1(shortcut.features)
        shortcut.features = self.act1(shortcut.features)

        shortcut2 = self.conv2(x)
        shortcut2.features = self.bn2(shortcut2.features)
        shortcut2.features = self.act2(shortcut2.features)

        shortcut3 = self.conv3(x)
        shortcut3.features = self.bn3(shortcut3.features)
        shortcut3.features = self.act3(shortcut3.features)
        shortcut.features = shortcut.features + \
            shortcut2.features + shortcut3.features

        shortcut.features = shortcut.features * x.features

        return shortcut


@MODELS.register_module()
class Asymm3DSpconv(BaseModule):
    """Asymmetrical 3D convolution networks.

    Args:
        grid_size (int): Size of voxel grids.
        input_dims (int): Input channels of the block.
        init_size (int): Initial size of feature channels before
            feeding into Encoder-Decoder structure. Default: 16.
        norm_cfg (dict): Dictionary to construct and config
            norm layer. Default: dict(type='BN1d', eps=1e-3, momentum=0.01)).
    """

    def __init__(self,
                 grid_size,
                 input_dims,
                 init_size=16,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)):
        super().__init__()

        self.grid_size = grid_size

        self.down_context = AsymmResBlock(
            input_dims, init_size, indice_key='pre', norm_cfg=norm_cfg)
        self.down_block0 = AsymmeDownBlock(
            init_size,
            2 * init_size,
            height_pooling=True,
            indice_key='down0',
            norm_cfg=norm_cfg)
        self.down_block1 = AsymmeDownBlock(
            2 * init_size,
            4 * init_size,
            height_pooling=True,
            indice_key='down1',
            norm_cfg=norm_cfg)
        self.down_block2 = AsymmeDownBlock(
            4 * init_size,
            8 * init_size,
            pooling=True,
            height_pooling=False,
            indice_key='down2',
            norm_cfg=norm_cfg)
        self.down_block3 = AsymmeDownBlock(
            8 * init_size,
            16 * init_size,
            pooling=True,
            height_pooling=False,
            indice_key='down3',
            norm_cfg=norm_cfg)

        self.up_block0 = AsymmeUpBlock(
            16 * init_size,
            16 * init_size,
            indice_key='up0',
            up_key='down3',
            norm_cfg=norm_cfg)
        self.up_block1 = AsymmeUpBlock(
            16 * init_size,
            8 * init_size,
            indice_key='up1',
            up_key='down2',
            norm_cfg=norm_cfg)
        self.up_block2 = AsymmeUpBlock(
            8 * init_size,
            4 * init_size,
            indice_key='up2',
            up_key='down1',
            norm_cfg=norm_cfg)
        self.up_block3 = AsymmeUpBlock(
            4 * init_size,
            2 * init_size,
            indice_key='up3',
            up_key='down0',
            norm_cfg=norm_cfg)

        self.ddcm = DDCMBlock(
            2 * init_size, 2 * init_size, indice_key='ddcm', norm_cfg=norm_cfg)

    def forward(self, voxel_features, coors, batch_size):
        """Forward pass."""
        coors = coors.int()
        ret = SparseConvTensor(voxel_features, coors, np.array(self.grid_size),
                               batch_size)
        ret = self.down_context(ret)
        down1_pool, down1_skip = self.down_block0(ret)
        down2_pool, down2_skip = self.down_block1(down1_pool)
        down3_pool, down3_skip = self.down_block2(down2_pool)
        down4_pool, down4_skip = self.down_block3(down3_pool)

        up4 = self.up_block0(down4_pool, down4_skip)
        up3 = self.up_block1(up4, down3_skip)
        up2 = self.up_block2(up3, down2_skip)
        up1 = self.up_block3(up2, down1_skip)

        up0 = self.ddcm(up1)

        up0.features = torch.cat((up0.features, up1.features), 1)

        return up0
