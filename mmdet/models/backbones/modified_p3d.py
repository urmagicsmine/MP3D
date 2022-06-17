import logging
import os
import os.path as osp
import torch
from collections import OrderedDict

import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn as mynn

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_state_dict
from mmcv.runner import BaseModule, ModuleList

from ..builder import BACKBONES
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from functools import partial

__all__ = ['modified_P3D', 'modified_P3D18', 'modified_P3D63', 'modified_P3D131','modified_P3D199', \
           'modified_P3D18ba', 'modified_P3D34ba']


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    rm_backbone=False):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    # load checkpoint from modelzoo or file or url
    if filename.startswith('modelzoo://') or \
            filename.startswith('torchvision://') or \
            filename.startswith('open-mmlab://') or \
            filename.startswith(('http://', 'https://')):
        raise NotImplementedError
    else:
        if not osp.isfile(filename):
            raise IOError('{} is not a checkpoint file'.format(filename))
        checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    if rm_backbone:  
        state_dict = {'.'.join(k.split('.')[1:]): v for k, v in checkpoint['state_dict'].items()}

    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        if rm_backbone:
            for k, v in checkpoint['state_dict'].items():
                print('Hi debug', k)
            state_dict = {'.'.join(k.split('.')[2:]): v for k, v in checkpoint['state_dict'].items()}
        else:
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    #print(state_dict.keys())
    # load state_dict
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint

def conv_S(in_planes,out_planes,stride=1,padding=1):
    # as is descriped, conv S is 1x3x3
    return nn.Conv3d(in_planes,out_planes,kernel_size=(1,3,3),stride=stride,
                     padding=padding,bias=False)

def conv_T(in_planes,out_planes,stride=1,padding=1):
    # conv T is 3x1x1
    return nn.Conv3d(in_planes,out_planes,kernel_size=(3,1,1),stride=stride,
                     padding=padding,bias=False)

def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 norm_cfg=dict(type='BN'),
                 n_s=0,
                 depth_3d=47,
                 ST_struc=('A','B','C'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None,
                 init_cfg=None
                 ):
        """Bottleneck block for P3D.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(BasicBlock, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert gcb is None or isinstance(gcb, dict)
        assert gen_attention is None or isinstance(gen_attention, dict)

        self.downsample = downsample
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = None
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.gcb = gcb
        self.with_gcb = gcb is not None
        self.gen_attention = gen_attention
        self.with_gen_attention = gen_attention is not None
        # for p3d parameters
        self.depth_3d=depth_3d
        self.ST_struc=ST_struc
        self.len_ST=len(self.ST_struc)
        assert n_s<self.depth_3d, "For detection backbone, all layers are conv3d in p3d, n_s should be smaller than depth_3d"
        stride_p=stride
        if not self.downsample == None:
            stride_p=(1,2,2)
        self.init_cfg=init_cfg

        if n_s==0:
            stride_p=1
        self.stride_p = stride_p
        self.id=n_s
        self.ST=list(self.ST_struc)[self.id%self.len_ST]

        # The first 3*3*3
        self.conv1 = conv_S(inplanes, planes, stride=stride_p, padding=(0,1,1))
        self.norm_1 = build_norm_layer(self.norm_cfg, planes)[1]
        #
        if self.ST == 'B':            
            self.conv2 = conv_T(inplanes, planes, stride=stride_p, padding=(1,0,0))
        else:
            self.conv2 = conv_T(planes, planes, stride=1, padding=(1,0,0))           
        self.norm_2 = build_norm_layer(self.norm_cfg, planes)[1]

        # The second 3*3*3 
        self.conv3 = conv_S(planes,planes, stride=1, padding=(0,1,1))
        self.norm_3 = build_norm_layer(self.norm_cfg, planes)[1]
        #
        self.conv4 = conv_T(planes,planes, stride=1, padding=(1,0,0))
        self.norm_4 = build_norm_layer(self.norm_cfg, planes)[1]

        self.relu = nn.ReLU(inplace=True)

        self.stride = stride

    def ST_A(self,x, convS, normS, convT, normT):
        x = convS(x)
        x = normS(x)
        x = self.relu(x)
        x = convT(x)
        x = normT(x)
        x = self.relu(x)

        return x

    def ST_B(self,x, convS, normS, convT, normT):
        tmp_x = convS(x)
        tmp_x = normS(tmp_x)
        tmp_x = self.relu(tmp_x)

        x = convT(x)
        x = normT(x)
        x = self.relu(x)

        return x+tmp_x

    def ST_C(self,x, convS, normS, convT, normT):
        x = convS(x)
        x = normS(x)
        x = self.relu(x)

        tmp_x = convT(x)
        tmp_x = normT(tmp_x)
        tmp_x = self.relu(tmp_x)

        return x+tmp_x

    def forward(self, x):
        residual = x
        if self.ST=='A':
            out=self.ST_A(x, self.conv1, self.norm_1, self.conv2, self.norm_2)
            out=self.ST_A(out, self.conv3, self.norm_3, self.conv4, self.norm_4)
        elif self.ST=='B':
            out=self.ST_B(x, self.conv1, self.norm_1, self.conv2, self.norm_2)
            out=self.ST_B(out, self.conv3, self.norm_3, self.conv4, self.norm_4)
        elif self.ST=='C':
            out=self.ST_C(x, self.conv1, self.norm_1, self.conv2, self.norm_2)
            out=self.ST_C(out, self.conv3, self.norm_3, self.conv4, self.norm_4)

        if self.downsample is not None:
            residual = self.downsample(x)
        #out += residual  #encounter inplace error
        out = out + residual
        out = self.relu(out)

        return out

class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 norm_cfg=dict(type='BN'),
                 n_s=0,
                 depth_3d=47,
                 ST_struc=('A','B','C'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None,
                 init_cfg=None
                 ):
        """Bottleneck block for P3D.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert gcb is None or isinstance(gcb, dict)
        assert gen_attention is None or isinstance(gen_attention, dict)
        
        self.downsample = downsample
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = None
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.gcb = gcb
        self.with_gcb = gcb is not None
        self.gen_attention = gen_attention
        self.with_gen_attention = gen_attention is not None
        self.init_cfg = init_cfg
        # for p3d parameters
        self.depth_3d=depth_3d
        self.ST_struc=ST_struc
        self.len_ST=len(self.ST_struc)
    
        stride_p=stride
        if not self.downsample ==None:
            stride_p=(1,2,2)
        if n_s<self.depth_3d:
            if n_s==0:
                stride_p=1
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False,stride=stride_p)

            self.norm_1 = build_norm_layer(self.norm_cfg, planes)[1]
            
        else:
            if n_s==self.depth_3d:
                stride_p=2
            else:
                stride_p=1
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,stride=stride_p)
            self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        self.id=n_s
        self.ST=list(self.ST_struc)[self.id%self.len_ST]
        if self.id<self.depth_3d:
            self.conv2 = conv_S(planes,planes, stride=1,padding=(0,1,1))
            self.norm_2 = build_norm_layer(self.norm_cfg, planes)[1]
            #
            self.conv3 = conv_T(planes,planes, stride=1,padding=(1,0,0))
            self.norm_3 = build_norm_layer(self.norm_cfg, planes)[1]
        else:
            self.conv_normal = nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1,bias=False)
            self.bn_normal = nn.BatchNorm2d(planes)

        if n_s<self.depth_3d:
            self.conv4 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
            self.norm_4 = build_norm_layer(self.norm_cfg, planes*4)[1]
        else:
            self.conv4 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn4 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        
    def ST_A(self,x):
        x = self.conv2(x)
        x = self.norm_2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm_3(x)
        x = self.relu(x)

        return x

    def ST_B(self,x):
        tmp_x = self.conv2(x)
        tmp_x = self.norm_2(tmp_x)
        tmp_x = self.relu(tmp_x)

        x = self.conv3(x)
        x = self.norm_3(x)
        x = self.relu(x)

        return x+tmp_x

    def ST_C(self,x):
        x = self.conv2(x)
        x = self.norm_2(x)
        x = self.relu(x)

        tmp_x = self.conv3(x)
        tmp_x = self.norm_3(tmp_x)
        tmp_x = self.relu(tmp_x)

        return x+tmp_x

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # print('it is x.shape after conv1',x.shape)
        # print('it is self.norm_1',self.norm_1)
        out = self.norm_1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        if self.id<self.depth_3d: # C3D parts: 

            if self.ST=='A':
                out=self.ST_A(out)
            elif self.ST=='B':
                out=self.ST_B(out)
            elif self.ST=='C':
                out=self.ST_C(out)
        else:
            out = self.conv_normal(out)   # normal is res5 part, C2D all.
            out = self.bn_normal(out)
            out = self.relu(out)

        out = self.conv4(out)
        out = self.norm_4(out)
        # print('it is residual before downsample ', residual.shape)
        # print('self.downsample',self.downsample)
        if self.downsample is not None:
            residual = self.downsample(x)
        # print('it is residual after downsample ', residual.shape)   
        # print('it is out.shape', out.shape)
        out += residual
        out = self.relu(out)

        return out



@BACKBONES.register_module
class modified_P3D(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    arch_settings = {
        18: (Bottleneck, (2, 2, 2, 2)),
        '18ba': (BasicBlock, (2, 2, 2, 2)),
        '34ba': (BasicBlock, (3, 4, 6, 3)),
        63: (Bottleneck, (3, 4, 6, 3)),
        131: (Bottleneck, (3, 4, 23, 3)),
        199: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 gcb=None,
                 stage_with_gcb=(False, False, False, False),
                 gen_attention=None,
                 stage_with_gen_attention=((), (), (), ()),
                 with_cp=False,
                 zero_init_residual=True,
                 modality='CT',
                 shortcut_type='B',
                 num_classes=400, # useness --cy
                 dropout=0.5,
                 ST_struc=('A','B','C'),
                 conversion_type='center_crop',
                 conversion_slice_num=9,
                 init_cfg=None):
        super(modified_P3D, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.gen_attention = gen_attention
        self.gcb = gcb
        self.stage_with_gcb = stage_with_gcb
        if gcb is not None:
            assert len(stage_with_gcb) == num_stages
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64
        self.init_cfg = init_cfg
        # p3d conv and layer 
        self.input_channel = 1 if modality=='CT' else 2  # 2 is for optical flow  --cy
        self.ST_struc=ST_struc
        self.conversion_type = conversion_type
        self.conversion_slice_num = conversion_slice_num
        self.conv1_custom = nn.Conv3d(self.input_channel, 64, kernel_size=(1,7,7), stride=(1,2,2),
                                padding=(0,3,3), bias=False)
        
        self.depth_3d=sum(stage_blocks[:num_stages])# C3D layers are only (res2,res3,res4),  res5 is C2D

        # self.bn1 = nn.BatchNorm3d(64) # bn1 is followed by conv1
        self.norm1_name, self.norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.cnt=0
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)       # pooling layer for conv1.

        self.layer1 = self._make_layer(self.block, 64, self.stage_blocks[0], shortcut_type)
        self.layer2 = self._make_layer(self.block, 128, self.stage_blocks[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.stage_blocks[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.stage_blocks[3], shortcut_type, stride=2)
        self.res_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        
        self.conversion_dict = {'center_crop': self.center_cropping}
        if self.conversion_type=='conv3d':
            self.conv3d_all_1 = nn.Conv3d(self.block.expansion*64, self.block.expansion*64, kernel_size=(self.conversion_slice_num, 1, 1), stride=(1, 1, 1),
                                          padding=(0, 0, 0), bias=False)
            self.conv3d_all_2 = nn.Conv3d(self.block.expansion*128, self.block.expansion*128, kernel_size=(self.conversion_slice_num, 1, 1), stride=(1, 1, 1),
                                          padding=(0, 0, 0), bias=False)
            self.conv3d_all_3 = nn.Conv3d(self.block.expansion*256, self.block.expansion*256, kernel_size=(self.conversion_slice_num, 1, 1), stride=(1, 1, 1),
                                          padding=(0, 0, 0), bias=False)
            self.conv3d_all_4 = nn.Conv3d(self.block.expansion*512, self.block.expansion*512, kernel_size=(self.conversion_slice_num, 1, 1), stride=(1, 1, 1),
                                          padding=(0, 0, 0), bias=False)
            self.conv3d_all_dict = {self.block.expansion*64:self.conv3d_all_1, \
                                    self.block.expansion*128:self.conv3d_all_2, \
                                    self.block.expansion*256:self.conv3d_all_3, \
                                    self.block.expansion*512:self.conv3d_all_4}
            self.conversion_dict['conv3d'] = self.conv3d_all
        elif self.conversion_type=='groupconv':
            self.groupconv_1 = nn.Conv2d(self.block.expansion*64 * self.conversion_slice_num, self.block.expansion*64, kernel_size=(1, 1), stride=(1, 1),
                                         padding=(0, 0), groups=self.block.expansion*64, bias=False)
            self.groupconv_2 = nn.Conv2d(self.block.expansion*128 * self.conversion_slice_num, self.block.expansion*128, kernel_size=(1, 1), stride=(1, 1),
                                         padding=(0, 0), groups=self.block.expansion*128, bias=False)
            self.groupconv_3 = nn.Conv2d(self.block.expansion*256 * self.conversion_slice_num, self.block.expansion*256, kernel_size=(1, 1), stride=(1, 1),
                                         padding=(0, 0), groups=self.block.expansion*256, bias=False)
            self.groupconv_4 = nn.Conv2d(self.block.expansion*512 * self.conversion_slice_num, self.block.expansion*512, kernel_size=(1, 1), stride=(1, 1),
                                         padding=(0, 0), groups=self.block.expansion*512, bias=False)
            self.group_conv_dict = {self.block.expansion*64:self.groupconv_1, \
                                    self.block.expansion*128:self.groupconv_2, \
                                    self.block.expansion*256:self.groupconv_3, \
                                    self.block.expansion*512:self.groupconv_4}
            self.conversion_dict['groupconv'] = self.group_conv
        elif self.conversion_type == 'keep3d':
            self.conversion_dict['keep3d'] = self.keep_3d
        self.avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=1)                              # pooling layer for res5.
        self.dropout=nn.Dropout(p=dropout)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # init_weight for p3d
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # # some private attribute
        # self.input_size=(self.input_channel,16,160,160)       # input of the network
        # self.input_mean = [0.485, 0.456, 0.406] if modality=='CT' else [0.5]
        # self.input_std = [0.229, 0.224, 0.225] if modality=='CT' else [np.mean([0.229, 0.224, 0.225])]

    #@property
    #def norm1(self):
        #return getattr(self, self.norm1_name)    
    
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        stride_p=stride #especially for downsample branch.

        if self.cnt<self.depth_3d:
            if self.cnt==0:
                stride_p=1
            else:
                stride_p=(1,2,2)
            #print('it is stride ', stride)
            #print('it is self.inplanes ', self.inplanes)
            #print('it is planes * block.expansion ', planes * block.expansion)
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A':
                    downsample = partial(downsample_basic_block,
                                         planes=planes * block.expansion,
                                         stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv3d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride_p, bias=False),
                        build_norm_layer(self.norm_cfg, planes * block.expansion)[1]
                    )
                    # print('it is downsample', downsample)

        else:
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A':
                    downsample = partial(downsample_basic_block,
                                         planes=planes * block.expansion,
                                         stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=2, bias=False),
                        build_norm_layer(self.norm_cfg, planes * block.expansion)[1]
                    )
        layers = []
        layers.append(block(self.inplanes, planes,  stride, norm_cfg=self.norm_cfg, downsample=downsample,n_s=self.cnt,depth_3d=self.depth_3d,ST_struc=self.ST_struc))
        self.cnt+=1

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_cfg=self.norm_cfg, n_s=self.cnt,depth_3d=self.depth_3d,ST_struc=self.ST_struc))
            self.cnt+=1

        return nn.Sequential(*layers)

    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1_custom, self.norm1]:
                 for param in m.parameters():
                         param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        pretrained = self.init_cfg.get('checkpoint', None)
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger, rm_backbone=True)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
                # add 3d conv init_weights and 3d norm init_weights
                elif isinstance(m, nn.Conv3d):
                    #print('here use Conv3d,initialize')
                    mynn.init.xavier_uniform(m.weight)
                elif isinstance(m, nn.BatchNorm3d):
                    #print('here use BatchNorm3d ')
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                    

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        # constant_init(m.norm3, 0)
                        continue
        else:
            raise TypeError('pretrained must be a str or None')


    
    def forward(self, x):
        x = self.conv1_custom(x)
        x = self.norm1(x)
        # print('it is self.norm1',self.norm1)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        # print('it is x shape before layer1',x.shape)
       # x = self.maxpool_2(self.layer1(x))  #  Part Res2
        x = self.layer1(x)
        center_x = self.conversion_dict[self.conversion_type](x)
        outs.append(center_x)
        x = self.layer2(x)
        center_x = self.conversion_dict[self.conversion_type](x)
        outs.append(center_x)
        x = self.layer3(x)
        center_x = self.conversion_dict[self.conversion_type](x)
        outs.append(center_x)
        sizes=x.size()
        # print('it is x before view',x.shape)
        x = self.layer4(x)
        # x = x.view(-1,sizes[1],sizes[3],sizes[4])  #  Part Res5
        # x = self.layer4(x)
        center_x = self.conversion_dict[self.conversion_type](x)
        outs.append(center_x)               # Part Res5
        
        
        return outs

    def train(self, mode=True):
        super(modified_P3D, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
    # add a function to get the center layer for CT image
    
    def center_cropping(self, x):
        """Center crop a N*C*D*H*W 3D feature map to N*C*H*W"""
        n, c, d, h, w = x.shape
        
        center_x = x[:,:,d//2,:,:]
        center_x = center_x.view(n, -1, h, w)
        # print('it is center_x',center_x.shape)
        return center_x
        
    def conv3d_all(self, x):
        n, c, d, h, w = x.shape
        out = self.conv3d_all_dict[c](x)
        out = out.view(n, -1, h, w)
        return out

    def group_conv(self, x):
        n, c, d, h, w = x.shape
        out = x
        out = out.view(n, c*d, h, w)
        out = self.group_conv_dict[c](out)
        return out

    def keep_3d(self, x):
        # if conversion_type == keep_3d, we won't chooce center slice feature,
        # but return origin 3d feature.
        return x
