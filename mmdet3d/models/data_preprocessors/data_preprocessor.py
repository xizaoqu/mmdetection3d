# Copyright (c) OpenMMLab. All rights reserved.
import math
from numbers import Number
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmdet.models import DetDataPreprocessor
from mmengine.model import stack_batch
from mmengine.utils import is_list_of
from torch.nn import functional as F

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import OptConfigType
from .utils import multiview_img_stack_batch
from .voxelize import VoxelizationByGridShape, dynamic_scatter_3d
import torch_scatter


@MODELS.register_module()
class Det3DDataPreprocessor(DetDataPreprocessor):
    """Points / Image pre-processor for point clouds / vision-only / multi-
    modality 3D detection tasks.

    It provides the data pre-processing as follows

    - Collate and move image and point cloud data to the target device.

    - 1) For image data:
    - Pad images in inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``.
    - Stack images in inputs to batch_imgs.
    - Convert images in inputs from bgr to rgb if the shape of input is
      (3, H, W).
    - Normalize images in inputs with defined std and mean.
    - Do batch augmentations during training.

    - 2) For point cloud data:
    - If no voxelization, directly return list of point cloud data.
    - If voxelization is applied, voxelize point cloud according to
      ``voxel_type`` and obtain ``voxels``.

    Args:
        voxel (bool): Whether to apply voxelization to point cloud.
            Defaults to False.
        voxel_type (str): Voxelization type. Two voxelization types are
            provided: 'hard' and 'dynamic', respectively for hard
            voxelization and dynamic voxelization. Defaults to 'hard'.
        voxel_layer (dict or :obj:`ConfigDict`, optional): Voxelization layer
            config. Defaults to None.
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic
            segmentation maps. Defaults to 255.
        bgr_to_rgb (bool): Whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): Whether to convert image from RGB to BGR.
            Defaults to False.
        boxtype2tensor (bool): Whether to keep the ``BaseBoxes`` type of
            bboxes data or not. Defaults to True.
        batch_augments (List[dict], optional): Batch-level augmentations.
            Defaults to None.
    """

    def __init__(self,
                 voxel: bool = False,
                 voxel_type: str = 'hard',
                 voxel_layer: OptConfigType = None,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 batch_augments: Optional[List[dict]] = None,
                 multi_sweep: bool=False) -> None:
        super(Det3DDataPreprocessor, self).__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            pad_mask=pad_mask,
            mask_pad_value=mask_pad_value,
            pad_seg=pad_seg,
            seg_pad_value=seg_pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            batch_augments=batch_augments)
        self.voxel = voxel
        self.voxel_type = voxel_type
        if voxel:
            self.voxel_layer = VoxelizationByGridShape(**voxel_layer)
        self.multi_sweep = multi_sweep

        if self.multi_sweep:
            from mmengine.fileio import load
            self.sweeps_infos = load('nuscenes_infos_10sweeps_train.pkl')

    def forward(self,
                data: Union[dict, List[dict]],
                training: bool = False) -> Union[dict, List[dict]]:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict or List[dict]): Data from dataloader.
                The dict contains the whole batch data, when it is
                a list[dict], the list indicate test time augmentation.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict or List[dict]: Data in the same format as the model input.
        """
        if isinstance(data, list):
            num_augs = len(data)
            aug_batch_data = []
            for aug_id in range(num_augs):
                single_aug_batch_data = self.simple_process(
                    data[aug_id], training)
                aug_batch_data.append(single_aug_batch_data)
            return aug_batch_data

        else:
            return self.simple_process(data, training)

    def simple_process(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion for img data
        based on ``BaseDataPreprocessor``, and voxelize point cloud if `voxel`
        is set to be True.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)

        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']

            if self.voxel:
                voxel_dict = self.voxelize(inputs['points'], data_samples)
                batch_inputs['voxels'] = voxel_dict

        if 'imgs' in inputs:
            imgs = inputs['imgs']

            if data_samples is not None:
                # NOTE the batched image size information may be useful, e.g.
                # in DETR, this is needed for the construction of masks, which
                # is then used for the transformer_head.
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples,
                                                  batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if hasattr(self, 'boxtype2tensor') and self.boxtype2tensor:
                    from mmdet.models.utils.misc import \
                        samplelist_boxtype2tensor
                    samplelist_boxtype2tensor(data_samples)
                elif hasattr(self, 'boxlist2tensor') and self.boxlist2tensor:
                    from mmdet.models.utils.misc import \
                        samplelist_boxlist2tensor
                    samplelist_boxlist2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)

                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs['imgs'] = imgs

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    def preprocess_img(self, _batch_img: torch.Tensor) -> torch.Tensor:
        # channel transform
        if self._channel_conversion:
            _batch_img = _batch_img[[2, 1, 0], ...]
        # Convert to float after channel conversion to ensure
        # efficiency
        _batch_img = _batch_img.float()
        # Normalization.
        if self._enable_normalize:
            if self.mean.shape[0] == 3:
                assert _batch_img.dim() == 3 and _batch_img.shape[0] == 3, (
                    'If the mean has 3 values, the input tensor '
                    'should in shape of (3, H, W), but got the '
                    f'tensor with shape {_batch_img.shape}')
            _batch_img = (_batch_img - self.mean) / self.std
        return _batch_img

    def collate_data(self, data: dict) -> dict:
        """Copying data to the target device and Performs normalization,
        padding and bgr2rgb conversion and stack based on
        ``BaseDataPreprocessor``.

        Collates the data sampled from dataloader into a list of dict and
        list of labels, and then copies tensor to the target device.

        Args:
            data (dict): Data sampled from dataloader.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore

        if 'img' in data['inputs']:
            _batch_imgs = data['inputs']['img']
            # Process data with `pseudo_collate`.
            if is_list_of(_batch_imgs, torch.Tensor):
                batch_imgs = []
                img_dim = _batch_imgs[0].dim()
                for _batch_img in _batch_imgs:
                    if img_dim == 3:  # standard img
                        _batch_img = self.preprocess_img(_batch_img)
                    elif img_dim == 4:
                        _batch_img = [
                            self.preprocess_img(_img) for _img in _batch_img
                        ]

                        _batch_img = torch.stack(_batch_img, dim=0)

                    batch_imgs.append(_batch_img)

                # Pad and stack Tensor.
                if img_dim == 3:
                    batch_imgs = stack_batch(batch_imgs, self.pad_size_divisor,
                                             self.pad_value)
                elif img_dim == 4:
                    batch_imgs = multiview_img_stack_batch(
                        batch_imgs, self.pad_size_divisor, self.pad_value)

            # Process data with `default_collate`.
            elif isinstance(_batch_imgs, torch.Tensor):
                assert _batch_imgs.dim() == 4, (
                    'The input of `ImgDataPreprocessor` should be a NCHW '
                    'tensor or a list of tensor, but got a tensor with '
                    f'shape: {_batch_imgs.shape}')
                if self._channel_conversion:
                    _batch_imgs = _batch_imgs[:, [2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_imgs = _batch_imgs.float()
                if self._enable_normalize:
                    _batch_imgs = (_batch_imgs - self.mean) / self.std
                h, w = _batch_imgs.shape[2:]
                target_h = math.ceil(
                    h / self.pad_size_divisor) * self.pad_size_divisor
                target_w = math.ceil(
                    w / self.pad_size_divisor) * self.pad_size_divisor
                pad_h = target_h - h
                pad_w = target_w - w
                batch_imgs = F.pad(_batch_imgs, (0, pad_w, 0, pad_h),
                                   'constant', self.pad_value)
            else:
                raise TypeError(
                    'Output of `cast_data` should be a list of dict '
                    'or a tuple with inputs and data_samples, but got'
                    f'{type(data)}: {data}')

            data['inputs']['imgs'] = batch_imgs

        data.setdefault('data_samples', None)

        return data

    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        # rewrite `_get_pad_shape` for obtaining image inputs.
        _batch_inputs = data['inputs']['img']
        # Process data with `pseudo_collate`.
        if is_list_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                if ori_input.dim() == 4:
                    # mean multiview input, select one of the
                    # image to calculate the pad shape
                    ori_input = ori_input[0]
                pad_h = int(
                    np.ceil(ori_input.shape[1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(ori_input.shape[2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = int(
                np.ceil(_batch_inputs.shape[1] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError('Output of `cast_data` should be a list of dict '
                            'or a tuple with inputs and data_samples, but got '
                            f'{type(data)}: {data}')
        return batch_pad_shape

    @torch.no_grad()
    def voxelize(self, points: List[torch.Tensor],
                 data_samples: SampleList) -> Dict[str, torch.Tensor]:
        """Apply voxelization to point cloud.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (list[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            Dict[str, Tensor]: Voxelization information.

            - voxels (Tensor): Features of voxels, shape is MxNxC for hard
              voxelization, NxC for dynamic voxelization.
            - coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
              where 1 represents the batch index.
            - num_points (Tensor, optional): Number of points in each voxel.
            - voxel_centers (Tensor, optional): Centers of voxels.
        """

        voxel_dict = dict()

        if self.voxel_type == 'hard':
            voxels, coors, num_points, voxel_centers = [], [], [], []
            for i, res in enumerate(points):
                res_voxels, res_coors, res_num_points = self.voxel_layer(res)
                res_voxel_centers = (
                    res_coors[:, [2, 1, 0]] + 0.5) * res_voxels.new_tensor(
                        self.voxel_layer.voxel_size) + res_voxels.new_tensor(
                            self.voxel_layer.point_cloud_range[0:3])
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                voxels.append(res_voxels)
                coors.append(res_coors)
                num_points.append(res_num_points)
                voxel_centers.append(res_voxel_centers)

            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)
            num_points = torch.cat(num_points, dim=0)
            voxel_centers = torch.cat(voxel_centers, dim=0)

            voxel_dict['num_points'] = num_points
            voxel_dict['voxel_centers'] = voxel_centers
        elif self.voxel_type == 'dynamic':
            coors = []
            # dynamic voxelization only provide a coors mapping
            for i, res in enumerate(points):
                res_coors = self.voxel_layer(res)
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                coors.append(res_coors)
            voxels = torch.cat(points, dim=0)
            coors = torch.cat(coors, dim=0)
        elif self.voxel_type == 'cylindrical':
            voxels, coors = [], []
            for i, (res, data_sample) in enumerate(zip(points, data_samples)):
                if self.multi_sweep:
                    res_mult_sweep = self.get_lidar_with_sweeps(res, self.sweeps_infos[data_sample.sample_idx]['sweeps'])
                    # for k in range(10):
                    #     pts = np.fromfile('data/nuscenes/sweeps/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin', dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
                rho = torch.sqrt(res[:, 0]**2 + res[:, 1]**2)
                phi = torch.atan2(res[:, 1], res[:, 0])
                polar_res = torch.stack((rho, phi, res[:, 2]), dim=-1)
                min_bound = polar_res.new_tensor(
                    self.voxel_layer.point_cloud_range[:3])
                max_bound = polar_res.new_tensor(
                    self.voxel_layer.point_cloud_range[3:])
                try:  # only support PyTorch >= 1.9.0  
                    polar_res_clamp = torch.clamp(polar_res, min_bound,
                                                  max_bound)
                    if self.multi_sweep:
                        rho_multi_sweep = torch.sqrt(res_mult_sweep[:, 0]**2 + res_mult_sweep[:, 1]**2)
                        phi_multi_sweep = torch.atan2(res_mult_sweep[:, 1], res_mult_sweep[:, 0])
                        polar_res_multi_sweep = torch.stack((rho_multi_sweep, phi_multi_sweep, res_mult_sweep[:, 2]), dim=-1)
                        polar_res_clamp_multi_sweep = torch.clamp(polar_res_multi_sweep, min_bound,
                                                    max_bound)
                except TypeError: # don't support multi_sweep now
                    polar_res_clamp = polar_res.clone()
                    for coor_idx in range(3):
                        polar_res_clamp[:, coor_idx][
                            polar_res[:, coor_idx] >
                            max_bound[coor_idx]] = max_bound[coor_idx]
                        polar_res_clamp[:, coor_idx][
                            polar_res[:, coor_idx] <
                            min_bound[coor_idx]] = min_bound[coor_idx]
                res_coors = torch.floor(
                    (polar_res_clamp - min_bound) / polar_res_clamp.new_tensor(
                        self.voxel_layer.voxel_size)).int()
                if self.multi_sweep:
                    res_coors_multi_sweep = torch.floor(
                        (polar_res_clamp_multi_sweep - min_bound) / polar_res_clamp_multi_sweep.new_tensor(
                            self.voxel_layer.voxel_size)).int()
                self.get_voxel_seg(res_coors, data_sample)
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                res_voxels = torch.cat((polar_res, res[:, :2], res[:, 3:]),
                                       dim=-1)
                voxels.append(res_voxels)
                coors.append(res_coors)
            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)
        elif self.voxel_type == 'minkunet':
            voxels, coors = [], []
            voxel_size = points[0].new_tensor(self.voxel_layer.voxel_size)
            for i, (res, data_sample) in enumerate(zip(points, data_samples)):
                res_coors = torch.round(res[:, :3] / voxel_size).int()
                res_coors -= res_coors.min(0)[0]

                res_coors_numpy = res_coors.cpu().numpy()
                inds, voxel2point_map = self.sparse_quantize(
                    res_coors_numpy, return_index=True, return_inverse=True)
                voxel2point_map = torch.from_numpy(voxel2point_map).cuda()
                if self.training:
                    if len(inds) > 80000:
                        inds = np.random.choice(inds, 80000, replace=False)
                inds = torch.from_numpy(inds).cuda()
                data_sample.gt_pts_seg.voxel_semantic_mask \
                    = data_sample.gt_pts_seg.pts_semantic_mask[inds]
                res_voxel_coors = res_coors[inds]
                res_voxels = res[inds]
                res_voxel_coors = F.pad(
                    res_voxel_coors, (0, 1), mode='constant', value=i)
                data_sample.voxel2point_map = voxel2point_map.long()
                voxels.append(res_voxels)
                coors.append(res_voxel_coors)
            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)

        else:
            raise ValueError(f'Invalid voxelization type {self.voxel_type}')

        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors

        return voxel_dict

    def get_voxel_seg(self, res_coors: torch.Tensor, data_sample: SampleList):
        """Get voxel-wise segmentation label and point2voxel map.

        Args:
            res_coors (Tensor): The voxel coordinates of points, Nx3.
            data_sample: (:obj:`Det3DDataSample`): The annotation data of
                every samples. Add voxel-wise annotation forsegmentation.
        """

        if self.training:
            if hasattr(data_sample.gt_pts_seg, 'pts_instance_mask'):
                pts_instance_mask = data_sample.gt_pts_seg.pts_instance_mask
                pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
                num_points = res_coors.shape[0]

                _, unique_indices_inverse = torch.unique(
                    res_coors, return_inverse=True, dim=0)

                unq_instance_labels = torch.unique(pts_instance_mask, dim=0)
                sort_instance_labels = pts_instance_mask.new_zeros(pts_instance_mask.shape)

                # map instance labels to [1 - the number of instances]
                i = 1
                ins2sem = pts_instance_mask.new_zeros(size=[len(unq_instance_labels)])
                for u in unq_instance_labels:
                    valid = pts_instance_mask == u
                    sort_instance_labels[valid] = i
                    ins2sem[i - 1] = pts_semantic_mask[valid][0]
                    i = i + 1

                flatten_inst_labels = res_coors.new_zeros(
                    (num_points, i))  # i: instance number+1(background)
                flatten_inst_labels[list(range(0, num_points)),
                                    sort_instance_labels[list(range(0, num_points))].
                                    cpu().numpy().tolist()] += 1
                # scatter point-wise labels into voxel-wise
                flatten_inst_label_sum = torch_scatter.scatter_sum(
                    flatten_inst_labels, unique_indices_inverse, dim=0)
                voxel_instance_labels = torch.argmax(flatten_inst_label_sum, dim=1)
                voxel_semantic_labels = ins2sem[voxel_instance_labels - 1]
                data_sample.gt_pts_seg.voxel_semantic_mask = voxel_semantic_labels
                data_sample.gt_pts_seg.voxel_instance_mask = voxel_instance_labels
            else:
                pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
                voxel_semantic_mask, _, point2voxel_map = dynamic_scatter_3d(
                    F.one_hot(pts_semantic_mask.long()).float(), res_coors, 'mean',
                    True)
                voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)
                data_sample.gt_pts_seg.voxel_semantic_mask = voxel_semantic_mask
                data_sample.gt_pts_seg.point2voxel_map = point2voxel_map

        else:
            pseudo_tensor = res_coors.new_ones([res_coors.shape[0], 1]).float()
            _, _, point2voxel_map = dynamic_scatter_3d(pseudo_tensor,
                                                       res_coors, 'mean', True)
            data_sample.gt_pts_seg.point2voxel_map = point2voxel_map

    def ravel_hash(self, x: np.ndarray) -> np.ndarray:
        """Get voxel coordinates hash for np.unique().

        Args:
            x (np.ndarray): The voxel coordinates of points, Nx3.

        Returns:
            np.ndarray: Voxels coordinates hash.
        """
        assert x.ndim == 2, x.shape

        x = x - np.min(x, axis=0)
        x = x.astype(np.uint64, copy=False)
        xmax = np.max(x, axis=0).astype(np.uint64) + 1

        h = np.zeros(x.shape[0], dtype=np.uint64)
        for k in range(x.shape[1] - 1):
            h += x[:, k]
            h *= xmax[k + 1]
        h += x[:, -1]
        return h

    def sparse_quantize(self,
                        coords: np.ndarray,
                        return_index: bool = False,
                        return_inverse: bool = False) -> List[np.ndarray]:
        """Sparse Quantization for voxel coordinates used in Minkunet.

        Args:
            coords (np.ndarray): The voxel coordinates of points, Nx3.
            return_index (bool): Whether to return the indices of the
                unique coords, shape (M,).
            return_inverse (bool): Whether to return the indices of the
                original coords shape (N,).

        Returns:
            List[np.ndarray] or None: Return index and inverse map if
            return_index and return_inverse is True.
        """
        _, indices, inverse_indices = np.unique(
            self.ravel_hash(coords), return_index=True, return_inverse=True)
        coords = coords[indices]

        outputs = []
        if return_index:
            outputs += [indices]
        if return_inverse:
            outputs += [inverse_indices]
        return outputs

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = sweep_info['lidar_path']
        points_sweep = np.fromfile('data/nuscenes/' + str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, points, sweeps, max_sweeps=5):
        sweep_points_list = [points]

        for k in np.random.choice(len(sweeps), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(sweeps[k])
            points_sweep = torch.from_numpy(points_sweep).to(points.device)
            sweep_points_list.append(points_sweep)

        points = torch.cat(sweep_points_list, axis=0)

        return points