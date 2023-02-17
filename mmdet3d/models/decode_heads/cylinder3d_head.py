# Copyright (c) OpenMMLab. All rights reserved.
import spconv
import torch
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from .decode_head import Base3DDecodeHead


@MODELS.register_module()
class Cylinder3DHead(Base3DDecodeHead):
    """"""

    def __init__(self,
                 channels,
                 num_classes,
                 dropout_ratio=0,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 loss_ce=dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 loss_lovasz=dict(type='mmseg.LovaszLoss', loss_weight=1.0),
                 ignore_index=0,
                 conv_seg_kernel_size=3,
                 init_cfg=None) -> None:
        super(Cylinder3DHead, self).__init__(
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            conv_seg_kernel_size=conv_seg_kernel_size,
            init_cfg=init_cfg)

        self.loss_lovasz = MODELS.build(loss_lovasz)
        self.loss_ce = MODELS.build(loss_ce)
        self.ignore_index = ignore_index

    def build_conv_seg(self, channels, num_classes, kernel_size):
        return spconv.SubMConv3d(
            channels,
            num_classes,
            indice_key='logit',
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=True)

    def forward(self, sparse_voxels: dict):
        """Forward function."""
        sparse_logits = self.logits(sparse_voxels)
        return sparse_logits

    def loss_by_feat(self, seg_logit: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute semantic segmentation loss.

        Args:
            seg_logit (torch.Tensor): Predicted per-point segmentation logits
                of shape [B, num_classes, N].
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_pts_seg`.
        """

        gt_semantic_segs = [
            data_sample.gt_pts_seg.voxel_semantic_mask
            for data_sample in batch_data_samples
        ]
        seg_label = torch.cat(gt_semantic_segs)
        seg_logit_feat = seg_logit.features
        loss = dict()
        loss['loss_ce'] = self.loss_ce(
            seg_logit_feat, seg_label, ignore_index=self.ignore_index)
        seg_logit_feat = seg_logit_feat.permute(1, 0)[None, :, :,
                                                      None]  # pseudo BCHW
        loss['loss_lovasz'] = self.loss_lovasz(
            seg_logit_feat, seg_label, ignore_index=self.ignore_index)

        return loss
