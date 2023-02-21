# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from ...structures.det3d_data_sample import SampleList
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
        encoded_feats = self.pts_voxel_encoder(
            batch_inputs['voxels']['voxels'], batch_inputs['voxels']['coors'])
        batch_inputs['voxels']['voxel_coors'] = encoded_feats[1]
        x = self.backbone(encoded_feats[0], encoded_feats[1],
                          len(batch_inputs['points']))
        if self.with_neck:
            x = self.neck(x)
        return x

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        # extract features using backbone
        x = self.extract_feat(batch_inputs_dict)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, batch_data_samples)
        losses.update(loss_decode)

        return losses

    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Simple test with single scene.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PixelData): Prediction of 3D semantic
              segmentation.
        """
        # 3D segmentation requires per-point prediction, so it's impossible
        # to use down-sampling to get a batch of scenes with same num_points
        # therefore, we only support testing one scene every time
        seg_pred_list = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)

        seg_logits = self.encode_decode(batch_inputs_dict, batch_input_metas)
        seg_predicts = seg_logits.features.argmax(1)
        coors = batch_inputs_dict['voxels']['voxel_coors']
        for batch_idx in range(len(batch_data_samples)):
            single_seg_predicts = seg_predicts[coors[:, 0] == batch_idx]
            point2voxel_map = batch_data_samples[
                batch_idx].gt_pts_seg.point2voxel_map.long()
            point_seg_predicts = single_seg_predicts[point2voxel_map]
            seg_pred_list.append(point_seg_predicts)

        return self.postprocess_result(seg_pred_list, batch_data_samples)
