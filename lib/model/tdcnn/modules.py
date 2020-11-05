import math

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_temporal_pooling.modules.roi_temporal_pool import _RoITemporalPooling
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss
from model.nl.fusion_modules import fusion_modules

DEBUG = False


class _TDCNN(nn.Module):
    """ faster RCNN """

    def __init__(self):
        super(_TDCNN, self).__init__()
        self.n_classes = cfg.NUM_CLASSES
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_twin = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_temporal_pool = _RoITemporalPooling(cfg.POOLING_LENGTH, cfg.POOLING_HEIGHT, cfg.POOLING_WIDTH,
                                                          cfg.DEDUP_TWINS)

        self.support_pool = nn.AvgPool3d(kernel_size=(1, 7, 7),
                                         stride=(1, 7, 7))
        self.cosine_pool = nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
        # nl
        self._fusion_modules = fusion_modules(in_channels=512, shot=self.shot)

    def prepare_data(self, video_data):
        return video_data

    def forward(self, video_data, gt_twins, support_data):
        batch_size = video_data.size(0)

        gt_twins = gt_twins.data
        # prepare data
        video_data = self.prepare_data(video_data)

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(video_data)
        rois, _, _, rpn_loss_cls, rpn_loss_twin, _, _ = self.RCNN_rpn(base_feat, gt_twins)  # torch.Size([1, 2000, 3])

        # if it is training phase, then use ground truth twins for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_twins)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_twin = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_temporal_pool(base_feat, rois.view(-1, 3))  # torch.Size([128, 512, 1, 4, 4])

        # pool support feature
        support_feature_list = []
        for i in range(self.shot):
            support_data[i] = self.prepare_data(support_data[i])  # torch.Size([1, 3, 768, 112, 112])
            tmp_support_feature = self.RCNN_base(support_data[i])  # torch.Size([1, 512, 96, 7, 7])
            tmp_support_feature = self.support_pool(tmp_support_feature)
            tmp_support_feature = torch.squeeze(tmp_support_feature)
            support_feature_list.append(tmp_support_feature)

        # non-local
        support_feature = support_feature_list[0]
        for i in range(1, self.shot):
            support_feature = torch.cat((support_feature, support_feature_list[i]), 1)

        if cfg.USE_ATTENTION:
            pooled_feat = self.RCNN_attention(pooled_feat)
            # feed pooled features to top model

        query_feature = self.cosine_pool(pooled_feat)
        query_feature = torch.squeeze(query_feature, 4)
        query_feature = torch.squeeze(query_feature, 3)
        query_feature = torch.squeeze(query_feature, 2)

        fusion_feature = self._fusion_modules(query_feature, support_feature.permute(1, 0).contiguous())

        pooled_feat = self._head_to_tail(fusion_feature) # torch.Size([128, 4096])
        # compute twin offset, twin_pred will be (128, 402)
        twin_pred = self.RCNN_twin_pred(pooled_feat)  # torch.Size([128, 4])

        if self.training:
            # select the corresponding columns according to roi labels, twin_pred will be (128, 2)
            twin_pred_view = twin_pred.view(twin_pred.size(0), int(twin_pred.size(1) / 2), 2)
            twin_pred_select = torch.gather(twin_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 2))
            twin_pred = twin_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, dim=1)

        RCNN_loss_cls = 0
        RCNN_loss_twin = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_twin = _smooth_l1_loss(twin_pred, rois_target, rois_inside_ws, rois_outside_ws)

            # RuntimeError caused by mGPUs and higher pytorch version: https://github.com/jwyang/faster-rcnn.pytorch/issues/226
            rpn_loss_cls = torch.unsqueeze(rpn_loss_cls, 0)
            rpn_loss_twin = torch.unsqueeze(rpn_loss_twin, 0)
            RCNN_loss_cls = torch.unsqueeze(RCNN_loss_cls, 0)
            RCNN_loss_twin = torch.unsqueeze(RCNN_loss_twin, 0)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        twin_pred = twin_pred.view(batch_size, rois.size(1), -1)

        if self.training:
            return rois, cls_prob, twin_pred, rpn_loss_cls, rpn_loss_twin, RCNN_loss_cls, RCNN_loss_twin, rois_label
        else:
            return rois, cls_prob, twin_pred

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        self.RCNN_rpn.init_weights()
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_twin_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    maxpool_count = 0
    for v in cfg:
        if v == 'M':
            maxpool_count += 1
            if maxpool_count==1:
                layers += [nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))]
            elif maxpool_count==5:
                layers += [nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=(0,1,1))]
            else:
                layers += [nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=(3,3,3), padding=(1,1,1))
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg_network = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

class C3D(nn.Module):
    """
    The C3D network as described in [1].
        References
        ----------
       [1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks."
       Proceedings of the IEEE international conference on computer vision. 2015.
    """

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def __init__(self):
        super(C3D, self).__init__()
        self.features = make_layers(cfg_network['A'], batch_norm=False)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(inplace=False),
            nn.Linear(4096, 487),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class c3d_tdcnn(_TDCNN):
    def __init__(self, pretrained=False, length_support=768, shot=0):
        self.model_path = 'data/pretrained_model/c3d-sports1m-pretrained.pth'
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.length_support = length_support
        self.shot = shot
        _TDCNN.__init__(self)

    def _init_modules(self):
        c3d = C3D()
        if self.pretrained:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            try:
                c3d.load_state_dict({k: v for k, v in state_dict.items() if (k in c3d.state_dict() and
                                                                             k != 'classifier.0.weight')})
            except Exception as e:
                print(e)

        # Using conv1 -> conv5b, not using the last maxpool
        self.RCNN_base = nn.Sequential(*list(c3d.features._modules.values())[:-1])
        # Using fc6
        self.RCNN_top = nn.Sequential(*list(c3d.classifier._modules.values())[:-4])
        # Fix the layers before pool2:
        for layer in range(6):
            for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
        self.RCNN_twin_pred = nn.Linear(4096, 2 * self.n_classes)

    def _head_to_tail(self, pool5_flat):
        fc6 = self.RCNN_top(pool5_flat)

        return fc6