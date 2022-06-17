import warnings

import torch

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class CE3D_Detector(TwoStageDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    #def __init__(self,
                 #backbone,
                 #neck=None,
                 #rpn_head=None,
                 #roi_head=None,
                 #train_cfg=None,
                 #test_cfg=None,
                 #pretrained=None,
                 #init_cfg=None):
        #super(CE3D_Detector, self).__init__(init_cfg)
    def __init__(self, in_channels=3, multi_view=False, *args, **kwargs):
        super(CE3D_Detector, self).__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.multi_view = multi_view

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        n, c, w, h = img.shape
        img = img.reshape(-1, self.in_channels, w, h)
        x = self.backbone(img)
        rpn_outs, rcnn_outs = self.neck(x)
        if self.multi_view:
            return rcnn_outs, rcnn_outs
        else:
            return rpn_outs, rcnn_outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        rpn_outs, rcnn_outs = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                rpn_outs,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(rcnn_outs, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        rpn_outs, rcnn_outs = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                rpn_outs, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            rcnn_outs, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        rpn_outs, rcnn_outs = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(rpn_outs, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            rcnn_outs, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        rpn_outs, rcnn_outs = self.extract_feat(img)
        proposal_list = self.rpn_head.aug_test_rpn(rpn_outs, img_metas)
        return self.roi_head.aug_test(
            rcnn_outs, proposal_list, img_metas, rescale=rescale)

