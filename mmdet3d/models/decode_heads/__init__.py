# Copyright (c) OpenMMLab. All rights reserved.
from .cylinder3d_head import Cylinder3DHead
from .dgcnn_head import DGCNNHead
from .minkunet_head import MinkUNetHead
from .paconv_head import PAConvHead
from .pointnet2_head import PointNet2Head
from .p3former_head import P3FormerHead

__all__ = [
    'PointNet2Head', 'DGCNNHead', 'PAConvHead', 'Cylinder3DHead',
    'MinkUNetHead', 'P3FormerHead'
]
