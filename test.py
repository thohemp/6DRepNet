from model import SixDRepNet
import math
import re
from matplotlib import pyplot as plt
import sys
import os
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.lib.function_base import _quantile_unchecked

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import datasets
import utils
import matplotlib
matplotlib.use('TkAgg')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--data_dir',
                        dest='data_dir', help='Directory path for data.',
                        default='datasets/AFLW2000', type=str)
    parser.add_argument('--filename_list',
                        dest='filename_list',
                        help='Path to text file containing relative paths for every example.',
                        default='datasets/AFLW2000/files.txt', type=str)  # datasets/BIWI_noTrack.npz
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument('--batch_size',
                        dest='batch_size', help='Batch size.',
                        default=64, type=int)
    parser.add_argument('--show_viz',
                        dest='show_viz', help='Save images with pose cube.',
                        default=False, type=bool)
    parser.add_argument('--dataset',
                        dest='dataset', help='Dataset type.',
                        default='AFLW2000', type=str)


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
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                        backbone_file='',
                        deploy=True,
                        pretrained=False)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(
                                              224), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    pose_dataset = datasets.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations, train_mode = False)
    test_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=args.batch_size,
        num_workers=2)


    # Load snapshot
    saved_state_dict = torch.load(snapshot_path, map_location='cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)    
    model.cuda(gpu)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    
    total = 0
    yaw_error = pitch_error = roll_error = .0
    v1_err = v2_err = v3_err = .0

    with torch.no_grad():

        for i, (images, r_label, cont_labels, name) in enumerate(test_loader):
            images = torch.Tensor(images).cuda(gpu)
            total += cont_labels.size(0)

            # gt matrix
            R_gt = r_label

            # gt euler
            y_gt_deg = cont_labels[:, 0].float()*180/np.pi
            p_gt_deg = cont_labels[:, 1].float()*180/np.pi
            r_gt_deg = cont_labels[:, 2].float()*180/np.pi

            R_pred = model(images)

            euler = utils.compute_euler_angles_from_rotation_matrices(
                R_pred)*180/np.pi
            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()

            R_pred = R_pred.cpu()
            v1_err += torch.sum(torch.acos(torch.clamp(
                torch.sum(R_gt[:, 0] * R_pred[:, 0], 1), -1, 1)) * 180/np.pi)
            v2_err += torch.sum(torch.acos(torch.clamp(
                torch.sum(R_gt[:, 1] * R_pred[:, 1], 1), -1, 1)) * 180/np.pi)
            v3_err += torch.sum(torch.acos(torch.clamp(
                torch.sum(R_gt[:, 2] * R_pred[:, 2], 1), -1, 1)) * 180/np.pi)

            pitch_error += torch.sum(torch.min(torch.stack((torch.abs(p_gt_deg - p_pred_deg), torch.abs(p_pred_deg + 360 - p_gt_deg), torch.abs(
                p_pred_deg - 360 - p_gt_deg), torch.abs(p_pred_deg + 180 - p_gt_deg), torch.abs(p_pred_deg - 180 - p_gt_deg))), 0)[0])
            yaw_error += torch.sum(torch.min(torch.stack((torch.abs(y_gt_deg - y_pred_deg), torch.abs(y_pred_deg + 360 - y_gt_deg), torch.abs(
                y_pred_deg - 360 - y_gt_deg), torch.abs(y_pred_deg + 180 - y_gt_deg), torch.abs(y_pred_deg - 180 - y_gt_deg))), 0)[0])
            roll_error += torch.sum(torch.min(torch.stack((torch.abs(r_gt_deg - r_pred_deg), torch.abs(r_pred_deg + 360 - r_gt_deg), torch.abs(
                r_pred_deg - 360 - r_gt_deg), torch.abs(r_pred_deg + 180 - r_gt_deg), torch.abs(r_pred_deg - 180 - r_gt_deg))), 0)[0])

            if args.show_viz:
                name = name[0]
                if args.dataset == 'AFLW2000':
                    cv2_img = cv2.imread(os.path.join(args.data_dir, name + '.jpg'))
                   
                elif args.dataset == 'BIWI':
                    vis = np.uint8(name)
                    h,w,c = vis.shape
                    vis2 = cv2.CreateMat(h, w, cv2.CV_32FC3)
                    vis0 = cv2.fromarray(vis)
                    cv2.CvtColor(vis0, vis2, cv2.CV_GRAY2BGR)
                    cv2_img = cv2.imread(vis2)
                utils.draw_axis(cv2_img, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], tdx=200, tdy=200, size=100)
                #utils.plot_pose_cube(cv2_img, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], size=200)
                cv2.imshow("Test", cv2_img)
                cv2.waitKey(5)
                cv2.imwrite(os.path.join('output/img/',name+'.png'),cv2_img)
                
        print('Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (
            yaw_error / total, pitch_error / total, roll_error / total,
            (yaw_error + pitch_error + roll_error) / (total * 3)))

        # print('Vec1: %.4f, Vec2: %.4f, Vec3: %.4f, VMAE: %.4f' % (
        #     v1_err / total, v2_err / total, v3_err / total,
        #     (v1_err + v2_err + v3_err) / (total * 3)))


