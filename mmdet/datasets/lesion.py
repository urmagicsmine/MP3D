# Copyright (c) OpenMMLab. All rights reserved.
import json
import contextlib
import io
import itertools
import logging
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from scipy import interpolate


from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset
from.coco import CocoDataset

@DATASETS.register_module()
class LesionDataset(CocoDataset):

    CLASSES = ('Lesion',)
    PALETTE = [(220, 20, 60),]

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {
            cat_id: i
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        json_file = json.load(open(ann_file, 'rb'))
        images = json_file['images']

        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            for j in images:
                if j['id'] == i:
                    info['slice_intv'] = j['slice_intv']
                    break
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def _parse_ann_info(self, img_info, ann_info, filter_small=False):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if filter_small:
                if ann['area'] <= 9 or w < 4 or h < 4:
                    continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann


    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            #iou_thrs = np.linspace(
            #     .1, 0.95, int(np.round((0.95 - .1) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        #===========
        # Eval FROC
        #===========
        avgFP = [0.5, 1, 2, 4]
        f_iou_th = 0.5
        all_boxes = []
        for cls in range(len(results[0])):
            tmp = [i[cls] for i in results]
            all_boxes.append(tmp)
        all_boxes.append(None)
        print('Evaluating on %.2f iou-thresh' % f_iou_th)
        self.eval_FROC(all_boxes, avgFP, f_iou_th)



        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results


    def get_gt_bboxes(self):
        gt_bboxes = [[np.empty((0,4), dtype=np.float32) for _ in self.img_ids] \
                for _ in range(len(self.CLASSES) + 1)]
        for idx, img_id in enumerate(self.img_ids):
            ann_info = self.get_ann_info(idx)
            for label in range(0, len(self.CLASSES)):
                box = ann_info['bboxes'][np.where(ann_info['labels']==label)[0]]
                gt_bboxes[label][idx] = box
        return gt_bboxes
    
    def eval_FROC(self, all_boxes, avgFP=[0.05, 0.1, 0.2, 0.5,1,2,3,4,8,16,32,64], iou_th=0.5):
        # all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
        # only one class for lesion dataset.
        # all_boxes[1][image] = N X 5
    
        gt_boxes = self.get_gt_bboxes()
        recall_per_class = [['0.0','-'] for i in range(1, len(all_boxes))]
        print('len all_boxes',len(all_boxes))
        for cls in range(0, len(all_boxes)-1):
            all_boxes, gt_boxes = all_boxes[cls], gt_boxes[cls]
            result, valid_avgFP, score_thresh = self.sens_at_FP(all_boxes, gt_boxes, avgFP, iou_th)
            print('='*20, 'cls:%s'%cls, '='*20)
            for idx, recall in enumerate(result[:2]):
                recall_per_class[cls-1][idx] = str(recall*100)[:4]
            for recall,fp, score in zip(result,valid_avgFP, score_thresh):
                print('Recall@%.2f=%.2f%%, score_threshold=%.4f' % (fp, recall*100, score))
            #TODO: when num of valid_avgFP < 6,is FROC correct?
            print('Mean FROC is %.2f'% np.mean(np.array(result[:6])*100))
        print('='*47)
        return recall_per_class

    def sens_at_FP(self, boxes_all, gts_all, avgFP, iou_th):
        # compute the sensitivity at avgFP (average FP per image)
        sens, fp_per_img, sorted_scores = self.FROC(boxes_all, gts_all, iou_th)
        max_fp = fp_per_img[-1]
        f = interpolate.interp1d(fp_per_img, sens, fill_value='extrapolate')
        s = interpolate.interp1d(sens, sorted_scores, fill_value='extrapolate')
        if(avgFP[-1] < max_fp):
            valid_avgFP_end_idx = len(avgFP)
        else:
            valid_avgFP_end_idx = np.argwhere(np.array(avgFP) > max_fp)[0][0]
        valid_avgFP = np.hstack((avgFP[:valid_avgFP_end_idx], max_fp))
        print(valid_avgFP)
        res = f(valid_avgFP)
        score_thresh = s(res)
        return res,valid_avgFP, score_thresh

    def FROC(self, boxes_all, gts_all, iou_th):
        # Compute the FROC curve, for single class only
        nImg = len(boxes_all)
        # img_idxs_ori : array([   0.,    0.,    0., ..., 4830., 4830., 4830.])
        img_idxs_ori = np.hstack([[i]*len(boxes_all[i]) for i in range(nImg)]).astype(int)
        boxes_cat = np.vstack(boxes_all)
        scores = boxes_cat[:, -1]
        ord = np.argsort(scores)[::-1]
        sorted_scores = scores[ord]
        boxes_cat = boxes_cat[ord, :4]
        img_idxs = img_idxs_ori[ord]
    
        hits = [np.zeros((len(gts),), dtype=bool) for gts in gts_all]
        nHits = 0
        nMiss = 0
        tps = []
        fps = []
        no_lesion = 0
        for i in range(len(boxes_cat)):
            overlaps = self.IOU(boxes_cat[i, :], gts_all[img_idxs[i]])
            if overlaps.shape[0] == 0:
                no_lesion += 1
                nMiss += 1
            elif overlaps.max() < iou_th:
                nMiss += 1
            else:
                for j in range(len(overlaps)):
                    if overlaps[j] >= iou_th and not hits[img_idxs[i]][j]:
                        hits[img_idxs[i]][j] = True
                        nHits += 1
    
            tps.append(nHits)
            fps.append(nMiss)
        nGt = len(np.vstack(gts_all))
        sens = np.array(tps, dtype=float) / nGt
        fp_per_img = np.array(fps, dtype=float) / nImg
        print('FROC:FP in no-lesion-images: ', no_lesion)
        return sens, fp_per_img, sorted_scores
    
    # In MMDet 2.x w = x2 - x1 instead of x2-x1+1
    @staticmethod
    def IOU(box1, gts):
        # compute overlaps
        # intersection
        ixmin = np.maximum(gts[:, 0], box1[0])
        iymin = np.maximum(gts[:, 1], box1[1])
        ixmax = np.minimum(gts[:, 2], box1[2])
        iymax = np.minimum(gts[:, 3], box1[3])
        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)
        inters = iw * ih
    
        # union
        uni = ((box1[2] - box1[0]) * (box1[3] - box1[1]) +
               (gts[:, 2] - gts[:, 0]) *
               (gts[:, 3] - gts[:, 1]) - inters)
    
        overlaps = inters / uni
        # ovmax = np.max(overlaps)
        # jmax = np.argmax(overlaps)
        return overlaps
