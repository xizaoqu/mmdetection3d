import re
from collections import OrderedDict

import torch
from mmengine.runner.checkpoint import _load_checkpoint

filename = 'checkpoints/model_save_backup.pt'
revise_keys = ([r'cylinder_3d_generator', 'voxel_encoder'], [
    r'cylinder_3d_spconv_seg', 'backbone'
], [
    r'voxel_encoder.fea_compression', 'pts_voxel_encoder.compression_layers'
], [r'voxel_encoder.PPmodel.0', 'pts_voxel_encoder.pre_norm'], [
    r'voxel_encoder.PPmodel.1.weight',
    'pts_voxel_encoder.vfe_layers.0.0.weight'
], [r'voxel_encoder.PPmodel.1.bias', 'pts_voxel_encoder.vfe_layers.0.0.bias'
    ], [r'voxel_encoder.PPmodel.2', 'pts_voxel_encoder.vfe_layers.0.1'], [
        r'voxel_encoder.PPmodel.4', 'pts_voxel_encoder.vfe_layers.1.0'
    ], [r'voxel_encoder.PPmodel.5', 'pts_voxel_encoder.vfe_layers.1.1'], [
        r'voxel_encoder.PPmodel.7', 'pts_voxel_encoder.vfe_layers.2.0'
    ], [r'voxel_encoder.PPmodel.8', 'pts_voxel_encoder.vfe_layers.2.1'
        ], [r'voxel_encoder.PPmodel.10', 'pts_voxel_encoder.vfe_layers.3'
            ], [r'backbone.downCntx', 'backbone.down_context'
                ], [r'backbone.resBlock2', 'backbone.down_block0'
                    ], [r'backbone.resBlock3', 'backbone.down_block1'
                        ], [r'backbone.resBlock4', 'backbone.down_block2'
                            ], [r'backbone.resBlock5', 'backbone.down_block3'],
               [r'backbone.upBlock0', 'backbone.up_block0'
                ], [r'backbone.upBlock1', 'backbone.up_block1'
                    ], [r'backbone.upBlock2', 'backbone.up_block2'
                        ], [r'backbone.upBlock3', 'backbone.up_block3'
                            ], [r'backbone.ReconNet',
                                'backbone.ddcm'], [r'conv1\.', 'conv0_0.'],
               [r'bn0\.', 'bn0_0.'], [r'act1\.',
                                      'act0_0.'], [r'conv1_2\.', 'conv0_1.'],
               [r'bn0_2\.',
                'bn0_1.'], [r'act1_2\.',
                            'act0_1.'], [r'conv2\.',
                                         'conv1_0.'], [r'act2\.', 'act1_0.'],
               [r'bn1\.',
                'bn1_0.'], [r'conv3\.',
                            'conv1_1.'], [r'act3\.',
                                          'act1_1.'], [r'bn2\.', 'bn1_1.'],
               [r'backbone.up_block0.conv0_0', 'backbone.up_block0.conv1'
                ], [r'backbone.up_block0.act0_0', 'backbone.up_block0.act1'
                    ], [r'backbone.up_block0.bn1_0', 'backbone.up_block0.bn1'],
               [r'backbone.up_block0.conv1_0', 'backbone.up_block0.conv2'
                ], [r'backbone.up_block0.act1_0', 'backbone.up_block0.act2'
                    ], [r'backbone.up_block0.bn1_1', 'backbone.up_block0.bn2'],
               [r'backbone.up_block0.conv1_1', 'backbone.up_block0.conv3'
                ], [r'backbone.up_block0.act1_1', 'backbone.up_block0.act3'], [
                    r'backbone.up_block1.conv0_0', 'backbone.up_block1.conv1'
                ], [r'backbone.up_block1.act0_0', 'backbone.up_block1.act1'
                    ], [r'backbone.up_block1.bn1_0', 'backbone.up_block1.bn1'],
               [r'backbone.up_block1.conv1_0', 'backbone.up_block1.conv2'], [
                   r'backbone.up_block1.act1_0', 'backbone.up_block1.act2'
               ], [r'backbone.up_block1.bn1_1', 'backbone.up_block1.bn2'], [
                   r'backbone.up_block1.conv1_1', 'backbone.up_block1.conv3'
               ], [r'backbone.up_block1.act1_1', 'backbone.up_block1.act3'], [
                   r'backbone.up_block2.conv0_0', 'backbone.up_block2.conv1'
               ], [r'backbone.up_block2.act0_0', 'backbone.up_block2.act1'], [
                   r'backbone.up_block2.bn1_0', 'backbone.up_block2.bn1'
               ], [r'backbone.up_block2.conv1_0', 'backbone.up_block2.conv2'],
               [r'backbone.up_block2.act1_0', 'backbone.up_block2.act2'
                ], [r'backbone.up_block2.bn1_1', 'backbone.up_block2.bn2'], [
                    r'backbone.up_block2.conv1_1', 'backbone.up_block2.conv3'
                ], [r'backbone.up_block2.act1_1', 'backbone.up_block2.act3'], [
                    r'backbone.up_block3.conv0_0', 'backbone.up_block3.conv1'
                ], [r'backbone.up_block3.act0_0', 'backbone.up_block3.act1'
                    ], [r'backbone.up_block3.bn1_0', 'backbone.up_block3.bn1'],
               [r'backbone.up_block3.conv1_0', 'backbone.up_block3.conv2'], [
                   r'backbone.up_block3.act1_0', 'backbone.up_block3.act2'
               ], [r'backbone.up_block3.bn1_1', 'backbone.up_block3.bn2'], [
                   r'backbone.up_block3.conv1_1', 'backbone.up_block3.conv3'
               ], [r'backbone.up_block3.act1_1', 'backbone.up_block3.act3'], [
                   r'backbone.ddcm.conv0_0', 'backbone.ddcm.conv1'
               ], [r'backbone.ddcm.bn0_0', 'backbone.ddcm.bn1'], [
                   r'backbone.ddcm.bn0_1', 'backbone.ddcm.bn2'
               ], [r'backbone.ddcm.conv0_1', 'backbone.ddcm.conv2'
                   ], [r'backbone.ddcm.conv1_3', 'backbone.ddcm.conv3'
                       ], [r'backbone.ddcm.bn0_3', 'backbone.ddcm.bn3'
                           ], [r'trans_dilao', 'trans_conv'
                               ], [r'backbone.logits', 'decode_head.conv_seg'])

checkpoint = _load_checkpoint(filename)
# OrderedDict is a subclass of dict
if not isinstance(checkpoint, dict):
    raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
# get state_dict from checkpoint
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# strip prefix of state_dict
metadata = getattr(state_dict, '_metadata', OrderedDict())
for p, r in revise_keys:
    state_dict = OrderedDict(
        {re.sub(p, r, k): v
         for k, v in state_dict.items()})

torch.save(state_dict, 'checkpoints/converted_model.pth')
