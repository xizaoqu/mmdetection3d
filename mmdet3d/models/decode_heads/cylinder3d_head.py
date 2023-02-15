# Copyright (c) OpenMMLab. All rights reserved.
import spconv
from decode_head import Base3DDecodeHead
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList


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
                 init_cfg=None) -> None:
        super(Cylinder3DHead, self).__init__(
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

        self.logits = spconv.SubMConv3d(
            channels,
            num_classes,
            indice_key='logit',
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.loss_lovasz = MODELS.build(loss_lovasz)
        self.loss_ce = MODELS.build(loss_ce)
        self.ignore_index = ignore_index

    def forward(self, feats_dict: dict):
        """Forward function."""
        sparse_voxels = feats_dict['sparse_voxels']
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
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        loss['loss_lovasz'] = self.loss_lovasz(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        loss['loss_ce'] = self.loss_ce(seg_logit, seg_label)

        return loss
