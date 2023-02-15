# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from .encoder_decoder import EncoderDecoder3D


@MODELS.register_module()
class Cylinder3D(EncoderDecoder3D):
    """"""

    def __init__(self,
                 pts_voxel_encoder: ConfigType,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 loss_regularization: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(Cylinder3D, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            loss_regularization=loss_regularization,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

    def extract_feat(self, batch_inputs: Tensor) -> Tensor:
        """Extract features from points."""
        encoded_feats = self.pts_voxel_encoder(batch_inputs)
        x = self.backbone(encoded_feats[0])
        if self.with_neck:
            x = self.neck(x)
        return x
