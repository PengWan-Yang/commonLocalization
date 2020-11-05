# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from pytorchgo.utils import logger
from pytorchgo.utils.pytorch_utils import model_summary, optimizer_summary
from model.tdcnn.modules import c3d_tdcnn
import json
import os,getpass
from model.rpn.twin_transform import twin_transform_inv
from model.rpn.twin_transform import clip_twins
from model.nms.nms_wrapper import nms

logger.auto_set_dir()

debug_small_iter  =  10
fast_eval_samples = 20

## args
learning_rate = 1e-5
len_train = 800
max_epoch = 36
decay_epoch = 24
gpus = 0
bs = 1
op = 'adam' #adam or sgd
tiou_thresholds=np.linspace(0.5, 0.95, 10)
max_per_video = 0
thresh = 0
test_nms = 0.4
length_support = 64 # length of support feature
shot = 1 # [1, 5]

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a R-C3D network')
    parser.add_argument('--dataset', dest='dataset',default='activitynet', type=str,
                      help='training dataset')
    parser.add_argument('--net', dest='net',default='c3d', type=str,
                      help='backbone')
    parser.add_argument('--start_epoch', dest='start_epoch', default=1, type=int,
                      help='starting epoch')
    parser.add_argument('--epochs', dest='max_epochs', default=max_epoch, type=int,
                      help='number of epochs to train')
    parser.add_argument('--disp_interval', default=100, type=int,
                      help='number of iterations to display')
    parser.add_argument('--nw', dest='num_workers', default=4, type=int,
                      help='number of worker to load data')
    parser.add_argument('--gpus', dest='gpus', nargs='+', type=int, default=gpus,
                      help='gpu ids.')                     
    parser.add_argument('--bs', dest='batch_size', default=bs, type=int,
                      help='batch_size')
    parser.add_argument('--roidb_dir', dest='roidb_dir',default="./preprocess",
                      help='roidb_dir')

    # config optimization
    parser.add_argument('--o', dest='optimizer',default=op, type=str,
                      help='training optimizer')
    parser.add_argument('--lr', dest='lr', default=learning_rate, type=float,
                      help='starting learning rate')
    parser.add_argument('--lr_decay_step', dest='lr_decay_step', default=decay_epoch, type=int,
                      help='step to do learning rate decay, unit is epoch')
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', default=0.1, type=float,
                      help='learning rate decay ratio')

    # resume trained model
    parser.add_argument('--test', action='store_true',
                        help='test')
    parser.add_argument('--is_debug', dest='is_debug', action='store_true',
                        help='debug')
    parser.add_argument('--len_train', dest='len_train', default=len_train, type=int,
                        help='len of train dataset loader')
    parser.add_argument('--max_per_video', dest='max_per_video', default=max_per_video, type=int,
                        help='max_per_video')
    parser.add_argument('--thresh', dest='thresh', default=thresh, type=float,
                        help='thresh of roi score')
    parser.add_argument('--shot', dest='shot', default=shot, type=int,
                        help='shot')

    parser.add_argument('--length_support', dest='length_support', default=length_support, type=int,
                        help='shot')
    parser.add_argument('--test_nms', dest='test_nms', default=test_nms, type=float,
                        help='test_nms')

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

def get_roidb(path):
    data = pickle.load(open(path, 'rb'))
    return data

def train_net(tdcnn_demo, dataloader, optimizer, args):
    # setting to train mode
    tdcnn_demo.train()
    loss_temp = 0

    for step, (support_data, video_data, gt_twins, num_gt) in tqdm(enumerate(dataloader),desc="training epoch {}/{}".format(args.epoch, args.max_epochs)):
        if is_debug and step > debug_small_iter:
            break

        video_data = video_data.cuda()
        for i in range(args.shot):
            support_data[i] = support_data[i].cuda()
        gt_twins = gt_twins.cuda()

        
        tdcnn_demo.zero_grad()
        rois, cls_prob, twin_pred, rpn_loss_cls, rpn_loss_twin, \
        RCNN_loss_cls, RCNN_loss_twin, rois_label = tdcnn_demo(video_data, gt_twins, support_data)
        loss = rpn_loss_cls.mean() + rpn_loss_twin.mean() + RCNN_loss_cls.mean() + RCNN_loss_twin.mean()
        loss_temp += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.disp_interval == 0:
            if step > 0:
                loss_temp /= args.disp_interval

            loss_rpn_cls = rpn_loss_cls.mean().item()
            loss_rpn_twin = rpn_loss_twin.mean().item()
            loss_rcnn_cls = RCNN_loss_cls.mean().item()
            loss_rcnn_twin = RCNN_loss_twin.mean().item()
            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt
            gt_cnt = num_gt.sum().item()

            logger.info("[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e, best_mAP: %.4f" \
                                    % (args.epoch, step+1, len(dataloader), loss_temp, args.lr, args.best_result))
            logger.info("fg/bg=(%d/%d), gt_twins: %d" % (fg_cnt, bg_cnt, gt_cnt, ))
            logger.info("rpn_cls: %.4f, rpn_twin: %.4f, rcnn_cls: %.4f, rcnn_twin %.4f" \
                          % (loss_rpn_cls, loss_rpn_twin, loss_rcnn_cls, loss_rcnn_twin))
            if args.best_loss > loss_temp:
                args.best_loss = loss_temp
                logger.info("best_loss: %.4f" % (loss_temp))
            loss_temp = 0



def nms_cpu(dets, thresh):
    dets = dets.numpy()
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    scores = dets[:, 2]

    length = (x2 - x1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)
        ovr = inter / (length[i] + length[order[1:]] - inter)

        inds = np.where(ovr < thresh)[0]
        order = order[inds]

    return torch.IntTensor(keep)

def test_net(tdcnn_demo, dataloader, args, split, max_per_video=0, thresh=0):
    np.random.seed(cfg.RNG_SEED)
    total_video_num = len(dataloader) * args.batch_size

    all_twins = [[[] for _ in range(total_video_num)]
                 for _ in range(args.num_classes)]  # class_num,video_num,proposal_num
    tdcnn_demo.eval()
    empty_array = np.transpose(np.array([[], [], []]), (1, 0))

    for data_idx, (support_data, video_data, gt_twins, num_gt, video_info) in tqdm(enumerate(dataloader),
                                                                                   desc="evaluation"):
        if is_debug and data_idx > fast_eval_samples:
            break

        video_data = video_data.cuda()
        for i in range(args.shot):
            support_data[i] = support_data[i].cuda()
        gt_twins = gt_twins.cuda()
        batch_size = video_data.shape[0]
        rois, cls_prob, twin_pred = tdcnn_demo(video_data, gt_twins,
                                               support_data)  ##torch.Size([1, 300, 3]),torch.Size([1, 300, 2]),torch.Size([1, 300, 4])
        scores_all = cls_prob.data
        twins = rois.data[:, :, 1:3]

        if cfg.TEST.TWIN_REG:  # True
            # Apply bounding-twin regression deltas
            twin_deltas = twin_pred.data
            if cfg.TRAIN.TWIN_NORMALIZE_TARGETS_PRECOMPUTED:  # True
                # Optionally normalize targets by a precomputed mean and stdev
                twin_deltas = twin_deltas.view(-1, 2) * torch.FloatTensor(cfg.TRAIN.TWIN_NORMALIZE_STDS).type_as(
                    twin_deltas) + torch.FloatTensor(cfg.TRAIN.TWIN_NORMALIZE_MEANS).type_as(twin_deltas)
                twin_deltas = twin_deltas.view(batch_size, -1, 2 * args.num_classes)  # torch.Size([1, 300, 4])

            pred_twins_all = twin_transform_inv(twins, twin_deltas, batch_size)  # torch.Size([1, 300, 4])
            pred_twins_all = clip_twins(pred_twins_all, cfg.TRAIN.LENGTH[0], batch_size)  # torch.Size([1, 300, 4])
        else:
            # Simply repeat the twins, once for each class
            pred_twins_all = np.tile(twins, (1, scores_all.shape[1]))

        for b in range(batch_size):
            if is_debug:
                logger.info(video_info)
            scores = scores_all[b]  # scores.squeeze()
            pred_twins = pred_twins_all[b]  # .squeeze()

            # skip j = 0, because it's the background class
            for j in range(1, args.num_classes):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    cls_twins = pred_twins[inds][:, j * 2:(j + 1) * 2]

                    cls_dets = torch.cat((cls_twins, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_twins, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    keep = nms_cpu(cls_dets.cpu(), args.test_nms)
                    if (len(keep) > 0):
                        if is_debug:
                            print("after nms, keep {}".format(len(keep)))
                        cls_dets = cls_dets[keep.view(-1).long()]
                    else:
                        print("warning, after nms, none of the rois is kept!!!")
                    all_twins[j][data_idx * batch_size + b] = cls_dets.cpu().numpy()
                else:
                    all_twins[j][data_idx * batch_size + b] = empty_array

            # Limit to max_per_video detections *over all classes*, useless code here, default max_per_video = 0
            if max_per_video > 0:
                video_scores = np.hstack([all_twins[j][data_idx * batch_size + b][:, -1]
                                          for j in range(1, args.num_classes)])
                if len(video_scores) > max_per_video:
                    video_thresh = np.sort(video_scores)[-max_per_video]
                    for j in range(1, args.num_classes):
                        keep = np.where(all_twins[j][data_idx * batch_size + b][:, -1] >= video_thresh)[0]
                        all_twins[j][data_idx * batch_size + b] = all_twins[j][data_idx * batch_size + b][keep, :]

            # logger.info('im_detect: {:d}/{:d}'.format(i * batch_size + b + 1, len(dataloader)))

    pred = dict()
    pred['external_data'] = ''
    pred['version'] = ''
    pred['results'] = dict()
    for i_video in tqdm(range(total_video_num), desc="generating prediction json.."):
        if is_debug and i_video > fast_eval_samples * batch_size - 2:
            break
        item_pre = []
        for j_roi in range(0, len(all_twins[1][
                                      i_video])):  # binary class problem, here we only consider class_num=1, ignoring background class
            _d = dict()
            _d['score'] = all_twins[1][i_video][j_roi][2].item()
            _d['label'] = 'c1'
            _d['segment'] = [all_twins[1][i_video][j_roi][0].item(),
                             all_twins[1][i_video][j_roi][1].item()]
            item_pre.append(_d)
        pred['results']["query_%05d" % i_video] = item_pre

    predict_filename = os.path.join(logger.get_logger_dir(), '{}_pred.json'.format(split))
    ground_truth_filename = os.path.join('preprocess/{}'.format(args.dataset), '{}_gt.json'.format(split))

    with open(predict_filename, 'w') as f:
        json.dump(pred, f)
        logger.info('dump pred.json complete..')

    sys.path.insert(0,"evaluation")
    from eval_detection import ANETdetection

    anet_detection = ANETdetection(ground_truth_filename, predict_filename ,
                                       subset="test", tiou_thresholds=tiou_thresholds,
                                       verbose=True, check_status=False)
    anet_detection.evaluate()
    ap = anet_detection.mAP
    mAP = ap[0]
    return mAP, ap


def do_test(logger, tdcnn_demo, dataloader_test):
    logger.info('do test')
    logger.info(args.test_nms)

    if torch.cuda.is_available():
        tdcnn_demo = tdcnn_demo.cuda()
        if isinstance(args.gpus, int):
            args.gpus = [args.gpus]
        tdcnn_demo = nn.parallel.DataParallel(tdcnn_demo, device_ids = args.gpus)

    state_dict = torch.load(os.path.join(logger.get_logger_dir(), "best_model.pth"))['model']
    logger.info("best_model.pth loaded!")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.{}'.format(k)
        if 'modules_focal' in k:
            k = k.replace('modules_focal', '_fusion_modules')
        new_state_dict[k] = v
    tdcnn_demo.load_state_dict(new_state_dict)
    tdcnn_demo.eval()

    test_mAP, test_ap = test_net(tdcnn_demo, dataloader=dataloader_test, args=args, split='test',
                                 max_per_video=args.max_per_video, thresh=args.thresh)
    tdcnn_demo.train()
    logger.info("final test set result: {}".format((test_mAP, test_ap)))
    logger.info("Congrats~")


    
if __name__ == '__main__':
    args = parse_args()
    assert args.shot >= 0

    is_debug = args.is_debug
    if is_debug:
        logger.info("is_debug=True!!!!!")



    logger.info('Called with args:')
    logger.info(args)

    args.imdb_name = "train_data.pkl"
    args.valdb_name = "val_data.pkl"
    args.testdb_name = "test_data.pkl"
    args.num_classes = 2

    if args.dataset == "activitynet":
        args.set_cfgs = ['ANCHOR_SCALES', '[1,1.25, 1.5,1.75, 2,2.5, 3,3.5, 4,4.5, 5,5.5, 6,7, 8,9,10,11,12,14,16,18,20,22,24,28,32,36,40,44,52,60,68,76,84,92,100]', 'NUM_CLASSES', args.num_classes]
    else:raise

    args.cfg_file = "cfgs/{}_{}.yml".format(args.net, args.dataset)

    cfg.CUDA = True
    cfg.USE_GPU_NMS = True

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    logger.info('Using config:')
    pprint.pprint(cfg)

    # for reproduce
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(cfg.RNG_SEED)

    cudnn.benchmark = True

    # train val test set
    roidb_path = os.path.join(args.roidb_dir,args.dataset,args.imdb_name)
    roidb = get_roidb(roidb_path)

    roidb_val_path = os.path.join(args.roidb_dir,args.dataset,args.valdb_name)
    roidb_val = get_roidb(roidb_val_path)

    roidb_test_path = os.path.join(args.roidb_dir,args.dataset, args.testdb_name)
    roidb_test = get_roidb(roidb_test_path)
    logger.info('{:d} roidb entries'.format(len(roidb)))

    dataset = roibatchLoader(roidb, phase='train', len_train=args.len_train, length_support=args.length_support, shot=args.shot)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=True)

    dataset_val = roibatchLoader(roidb_val, phase='val', length_support=args.length_support, shot=args.shot)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size,
                                             num_workers=args.num_workers, shuffle=False)

    dataset_test = roibatchLoader(roidb_test, phase='test', length_support=args.length_support, shot=args.shot)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                             num_workers=args.num_workers, shuffle=False)

    # initilize the network here.
    if args.net == 'c3d':
        tdcnn_demo = c3d_tdcnn(pretrained=True, length_support=args.length_support,
                            shot=args.shot)
    else:
        logger.info("network is not defined");raise

    tdcnn_demo.create_architecture()


    if args.test:
        do_test(logger=logger, tdcnn_demo=tdcnn_demo, dataloader_test=dataloader_test)
        exit(0)

    params = []
    for key, value in dict(tdcnn_demo.named_parameters()).items():
        if value.requires_grad:
            logger.info(key)
            if 'bias' in key:
                params += [{'params':[value],'lr': args.lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                    'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params':[value],'lr': args.lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise


    if torch.cuda.is_available():
        tdcnn_demo = tdcnn_demo.cuda()
        if isinstance(args.gpus, int):
            args.gpus = [args.gpus]
        tdcnn_demo = nn.parallel.DataParallel(tdcnn_demo, device_ids = args.gpus)

    model_summary([tdcnn_demo])
    optimizer_summary(optimizer)

    args.best_result = -1
    args.best_loss = 1000
    for epoch in range(args.start_epoch, args.max_epochs):
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            args.lr *= args.lr_decay_gamma
            
        args.epoch = epoch
        train_net(tdcnn_demo, dataloader, optimizer, args)

        tdcnn_demo.eval()
        test_mAP, test_ap = test_net(tdcnn_demo, dataloader=dataloader_test, args=args, split='test',
                                     max_per_video=args.max_per_video, thresh=args.thresh)
        tdcnn_demo.train()#recover for training mode

        logger.info("current result: {},{}".format(test_mAP, test_ap))
        if test_mAP > args.best_result:
            logger.info("current result {} better than {}, save best_model.".format(test_mAP, args.best_result))
            args.best_result = test_mAP
            save_checkpoint({
                'model': tdcnn_demo.module.state_dict() if len(args.gpus) > 1 else tdcnn_demo.state_dict(),
                'best': args.best_result,
            }, os.path.join(logger.get_logger_dir(), 'best_model.pth'))

    # reload the best weight, do final testing
    state_dict = torch.load(os.path.join(logger.get_logger_dir(), 'best_model.pth'))['model']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.{}'.format(k)
        new_state_dict[k] = v
    tdcnn_demo.load_state_dict(new_state_dict)
    tdcnn_demo.eval()

    test_mAP, test_ap = test_net(tdcnn_demo, dataloader=dataloader_test, args=args, split='test',
                                 max_per_video=args.max_per_video, thresh=args.thresh)
    logger.info("final test set result: {},{}".format(test_mAP, test_ap))

    logger.info("Congrats~")



