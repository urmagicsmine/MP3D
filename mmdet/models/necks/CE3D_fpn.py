import warnings

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS
from .fpn import FPN


@NECKS.register_module()
class CE3D_FPN(FPN):
    """
    Args:
        num_images_3dce: num of 3-channel used in 3dce. if set to 3, then use 3*3=9 2D slices as input.
        is_conv1x1: whether to apply a 1x1 conv layer after rcnn_outs to restore the num_channels of
            features which is sent into RCNN. If False, in_channels of RCNN should be set as
            'NUM_IMAGES_3DCE * in_channels.'
    """

    #def __init__(self,
                 #in_channels=None,
                 #out_channels=None,
                 #num_outs=None,
                 #is_conv1x1=True,
                 #num_images_3dce=3,
                 #start_level=0,
                 #end_level=-1,
                 #add_extra_convs=False,
                 #extra_convs_on_inputs=True,
                 #relu_before_extra_convs=False,
                 #no_norm_on_lateral=False,
                 #conv_cfg=None,
                 #norm_cfg=None,
                 #act_cfg=None,
                 #upsample_cfg=dict(mode='nearest'),
                 #init_cfg=dict(
                     #type='Xavier', layer='Conv2d', distribution='uniform')):
        #super(CE3D_FPN, self).__init__()
    def __init__(self, in_channels, out_channels, is_conv1x1=True, num_images_3dce=3, **kwargs):
        super(CE3D_FPN, self).__init__(in_channels, out_channels, **kwargs)
        #assert isinstance(in_channels, list)
        #self.in_channels = in_channels
        #self.out_channels = out_channels
        #self.num_ins = len(in_channels)
        #self.num_outs = num_outs
        self.is_conv1x1 = is_conv1x1
        self.num_images_3dce = num_images_3dce
        if self.is_conv1x1:
            self.fuse_conv = ConvModule(self.num_images_3dce*out_channels, out_channels, 1)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        outs_rpn = []
        outs_rcnn = []
        for out in outs:
            n, c, h, w = out.shape
            M = self.num_images_3dce
            out = out.view(n // M, M * c, h, w)
            outs_rpn.append(out[:, (M // 2) * c: (M // 2 + 1) * c, :, :])
            if self.is_conv1x1:
                out = self.fuse_conv(out)
            outs_rcnn.append(out)

        return tuple(outs_rpn), tuple(outs_rcnn)

