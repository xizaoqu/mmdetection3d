# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet3d.registry import MODELS


def test_cylinder3d():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    voxel_feats = torch.rand(50, 16).cuda()
    coorx = torch.randint(0, 480, (50, 1)).cuda()
    coory = torch.randint(0, 360, (50, 1)).cuda()
    coorz = torch.randint(0, 32, (50, 1)).cuda()
    coorbatch0 = torch.zeros(50, 1).cuda()
    coors = torch.cat([coorbatch0, coorx, coory, coorz], dim=1)
    batch_size = 1

    cfg = dict(
        type='Asymm3DSpconv',
        grid_size=[480, 360, 32],
        input_dims=16,
        init_size=32,
    )

    self = MODELS.build(cfg).cuda()
    self.init_weights()

    y = self(voxel_feats, coors, batch_size)
    assert y.batch_size == 1
    assert y.features.shape == torch.Size([50, 128])
    assert y.indices.shape == torch.Size([50, 4])
