
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import torch

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch

import numpy as np
import random
import time
import pdb
import json


class roibatchLoader(data.Dataset):
    def __init__(self, roidb, normalize=None, phase='train', len_train=2000, length_support=768, shot=0, step_frame=1):
        self._roidb = roidb
        self.max_num_box = cfg.MAX_NUM_GT_TWINS#20
        self.normalize = normalize
        self.phase = phase
        self.step_frame = step_frame
        self.len_train = len_train
        self.length_support = length_support
        self.shot = shot
        if phase == 'train':
            classes = self._roidb.keys()
            self.classes = list(sorted(classes))
    def __getitem__(self, index):
        # get the anchor index for current sample index
        if self.phase == 'train':
            classes = self.classes
            len_class = len(classes)
            index_class = np.random.randint(0, high=len_class)
            segments_class = self._roidb[classes[index_class]]
            len_segment = len(segments_class)
            index_segment = np.random.randint(0, high=len_segment, size=(1+self.shot))

            data_support = []
            if self.shot > 0:
                for i in range(self.shot):
                    item_support = segments_class[index_segment[i]]
                    start = item_support['frames'][0][1]
                    gt_start = item_support['wins'][0][0] + start
                    gt_end = item_support['wins'][0][1] + start
                    item_support['frames'][0][1] = gt_start
                    item_support['frames'][0][2] = gt_end  # only keep foreground, remove all background

                    blobs_support = get_minibatch([item_support], self.phase, self.step_frame,
                                                  length_support=self.length_support)
                    tmp_support = torch.from_numpy(blobs_support['data'])
                    length, height, width = tmp_support.shape[-3:]
                    tmp_support = tmp_support.contiguous().view(3, length, height, width)
                    data_support.append(tmp_support)


            # query
            item_query = segments_class[index_segment[self.shot]]
            blobs_query = get_minibatch([item_query], self.phase, self.step_frame)
            data_query = torch.from_numpy(blobs_query['data'])
            length, height, width = data_query.shape[-3:]
            data_query = data_query.contiguous().view(3, length, height, width)

            gt_windows = torch.from_numpy(blobs_query['gt_windows'])
            gt_windows_padding = gt_windows.new(self.max_num_box, gt_windows.size(1)).zero_()
            num_gt = min(gt_windows.size(0), self.max_num_box)
            gt_windows_padding[:num_gt, :] = gt_windows[:num_gt]
            return data_support, data_query, gt_windows_padding, num_gt
        else:
            data_support = []
            if self.shot > 0:
                for i in range(1, (self.shot + 1)):

                    item_support = self._roidb[index][i % 5]

                    blobs_support = get_minibatch([item_support], self.phase, self.step_frame,
                                                  length_support=self.length_support)
                    tmp_support = torch.from_numpy(blobs_support['data'])
                    length, height, width = tmp_support.shape[-3:]
                    tmp_support = tmp_support.contiguous().view(3, length, height, width) #support video is already cropped in pickle file, we here don't need to crop it again.
                    data_support.append(tmp_support)

            item_query = self._roidb[index][0]
            blobs_query = get_minibatch([item_query], self.phase, self.step_frame)
            data_query = torch.from_numpy(blobs_query['data'])
            length, height, width = data_query.shape[-3:]
            data_query = data_query.contiguous().view(3, length, height, width)

            gt_windows = torch.from_numpy(blobs_query['gt_windows'])
            gt_windows_padding = gt_windows.new(self.max_num_box, gt_windows.size(1)).zero_()
            num_gt = min(gt_windows.size(0), self.max_num_box)
            gt_windows_padding[:num_gt, :] = gt_windows[:num_gt]

            video_info = item_query['video_id']
            return data_support, data_query, gt_windows_padding, num_gt, video_info

    def __len__(self):
        if self.phase == 'train':
            return self.len_train
        else:
            return len(self._roidb)



