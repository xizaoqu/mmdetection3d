# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List, Optional, Sequence

from mmengine.logging import MMLogger
import mmengine
from mmdet3d.evaluation import panoptic_seg_eval
from mmdet3d.registry import METRICS
from .seg_metric import SegMetric
import os
import json
import numpy as np

@METRICS.register_module()
class PanopticSegMetric(SegMetric):
    """3D Panoptic segmentation evaluation metric.

    Args:
        thing_class_inds (list[int]): Indices of thing classes.
        stuff_class_inds (list[int]): Indices of stuff classes.
        min_num_points (int): Minimum number of points of an object to be
            counted as ground truth in evaluation.
        id_offset (int): Offset for instance ids to concat with
            semantic labels.
        dataset_type (str): Type of dataset.
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default to None.
        pklfile_prefix (str, optional): The prefix of pkl files, including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default to None.
        submission_prefix (str, optional): The prefix of submission data.
            If not specified, the submission data will not be generated.
            Default to None.
    """

    def __init__(self,
                 thing_class_inds: List[int],
                 stuff_class_inds: List[int],
                 min_num_points: int,
                 id_offset: int,
                 dataset_type: str,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 taskset: str = 'None',
                 **kwargs):
        self.thing_class_inds = thing_class_inds
        self.stuff_class_inds = stuff_class_inds
        self.min_num_points = min_num_points
        self.id_offset = id_offset
        self.dataset_type=dataset_type
        self.taskset = taskset

        super(PanopticSegMetric, self).__init__(
            pklfile_prefix=pklfile_prefix,
            submission_prefix=submission_prefix,
            prefix=prefix,
            collect_device=collect_device,
            **kwargs)

    # TODO modify format_result for panoptic segmentation evaluation, \
    # different datasets have different needs.

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix:
            self.format_results(results)
            return None

        label2cat = self.dataset_meta['label2cat']
        ignore_index = self.dataset_meta['ignore_index']
        classes = self.dataset_meta['classes']
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        include = self.thing_class_inds + self.stuff_class_inds

        gt_labels = []
        seg_preds = []
        for eval_ann, sinlge_pred_results in results:
            gt_labels.append(eval_ann)
            seg_preds.append(sinlge_pred_results)

        ret_dict = panoptic_seg_eval(gt_labels, seg_preds, classes,
                                     thing_classes, stuff_classes, include, self.dataset_type,
                                     self.min_num_points, self.id_offset,
                                     label2cat, ignore_index, logger)

        return ret_dict

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            pred_3d = data_sample['pred_pts_seg']
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu').numpy()
                else:
                    cpu_pred_3d[k] = v
            self.results.append((eval_ann_info, cpu_pred_3d))

    def format_results(self, results):
        r"""Format the results to txt file. Refer to `ScanNet documentation
        <http://kaldir.vc.in.tum.de/scannet_benchmark/documentation>`_.

        Args:
            outputs (list[dict]): Testing results of the dataset.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving submission
                files when ``submission_prefix`` is not specified.
        """

        submission_prefix = self.submission_prefix
        mmengine.mkdir_or_exist(submission_prefix)
        ignore_index = self.dataset_meta['ignore_index']


        if self.dataset_type == 'nuscenes':
            meta_dir = os.path.join(submission_prefix, self.taskset)
            mmengine.mkdir_or_exist(meta_dir)
            meta =  {"meta": {
                "task": "segmentation",
                "use_camera": False,
                "use_lidar": True,
                "use_radar": False,
                "use_map": False,
                "use_external": False}}
            output = open(os.path.join(submission_prefix, self.taskset, 'submission.json'), 'w')
            json_meta = json.dumps(meta)
            output.write(json_meta)
            output.close()

            for i, (eval_ann, result) in enumerate(results):
                sample_token = eval_ann['token']
                pred_file_dir = os.path.join(submission_prefix, 'panoptic', self.taskset, sample_token)
                mmengine.mkdir_or_exist(pred_file_dir)
                pred_semantic_mask = result['pts_semantic_mask']
                pred_instance_mask = result['pts_instance_mask']
                pred_panoptic_mask = (pred_instance_mask + pred_semantic_mask*self.id_offset).astype(np.uint16)
                curr_file = os.path.join(pred_file_dir,  "_panoptic.npz")
                np.savez_compressed(curr_file, data=pred_panoptic_mask)
        elif self.dataset_type == "semantickitti":
            pass