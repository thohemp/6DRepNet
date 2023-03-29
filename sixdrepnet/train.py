import time
import math
import re
import sys
import os
import argparse

import numpy as np
from numpy.lib.function_base import _quantile_unchecked
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.utils import model_zoo
import torchvision
from torchvision import transforms
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
matplotlib.use('TkAgg')

from model import SixDRepNet, SixDRepNet2
import datasets
from loss import GeodesicLoss


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs',
        help='Maximum number of training epochs.',
        default=80, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=80, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.0001, type=float)
    parser.add_argument('--scheduler', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type.',
        default='Pose_300W_LP', type=str) #Pose_300W_LP
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='datasets/300W_LP', type=str)#BIWI_70_30_train.npz
    parser.add_argument(
        '--filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='datasets/300W_LP/files.txt', type=str) #BIWI_70_30_train.npz #300W_LP/files.txt
    parser.add_argument(
        '--output_string', dest='output_string',
        help='String appended to output snapshots.', default='', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)

    args = parser.parse_args()
    return args

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


if __name__ == '__main__':

    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    b_scheduler = args.scheduler

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    summary_name = '{}_{}_bs{}'.format(
        'SixDRepNet', int(time.time()), args.batch_size)

    if not os.path.exists('output/snapshots/{}'.format(summary_name)):
        os.makedirs('output/snapshots/{}'.format(summary_name))

    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                        backbone_file='RepVGG-B1g2-train.pth',
                        deploy=False,
                        pretrained=True)
 
    if not args.snapshot == '':
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict['model_state_dict'])

    print('Loading data.')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    transformations = transforms.Compose([transforms.RandomResizedCrop(size=224,scale=(0.8,1)),
                                          transforms.ToTensor(),
                                          normalize])

    pose_dataset = datasets.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations)

    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    model.cuda(gpu)
    crit = GeodesicLoss().cuda(gpu) #torch.nn.MSELoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)


    if not args.snapshot == '':
        optimizer.load_state_dict(saved_state_dict['optimizer_state_dict'])

    #milestones = np.arange(num_epochs)
    milestones = [10, 20]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)

    print('Starting training.')
    for epoch in range(num_epochs):
        loss_sum = .0
        iter = 0
        for i, (images, gt_mat, _, _) in enumerate(train_loader):
            iter += 1
            images = torch.Tensor(images).cuda(gpu)

            # Forward pass
            pred_mat = model(images)

            # Calc loss
            loss = crit(gt_mat.cuda(gpu), pred_mat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: '
                      '%.6f' % (
                          epoch+1,
                          num_epochs,
                          i+1,
                          len(pose_dataset)//batch_size,
                          loss.item(),
                      )
                      )
        
        if b_scheduler:
            scheduler.step()

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...',
                  torch.save({
                      'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                  }, 'output/snapshots/' + summary_name + '/' + args.output_string +
                      '_epoch_' + str(epoch+1) + '.tar')
                  )
