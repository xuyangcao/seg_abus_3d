import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import math
import argparse
from tqdm import tqdm 
import numpy as np
from collections import OrderedDict
import h5py
import nibabel as nib
from medpy import metric
import pandas as pd

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from networks.vnet import VNet
from utils.test_util import test_all_case
from dataloaders.abus import ABUS, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler 

def get_args():
    fold = 2
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./data/abus_roi/', help='data root path')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--start_epoch', type=int, default=40000)

    parser.add_argument('--snapshot_path', type=str, default='./work/0110_dice_'+str(fold), help='snapshot path')
    parser.add_argument('--test_save_path', type=str, default='./results/test_'+str(fold), help='save path')
    parser.add_argument('--fold', type=str,  default=str(fold), help='random seed')

    parser.add_argument('--use_tm', action='store_true', default=False, help='whether use threshold_map')

    args = parser.parse_args()
    return args


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd

def transpose(image):
    image = np.transpose(image, (2, 0, 1))
    return image


def test(args, net, testloader, num_classes, save_result=True, test_save_path=None):
    total_metric = 0.0
    metric_dict = OrderedDict()
    metric_dict['name'] = list()
    metric_dict['dice'] = list()
    metric_dict['jaccard'] = list()
    metric_dict['asd'] = list()
    metric_dict['95hd'] = list()

    with torch.no_grad():
        for sample in tqdm(testloader): 
            image = sample['image']
            label = sample['label']
            case_name = sample['filename']
            image, label = image.cuda(), label.cuda()
            out = net(image)
            score_map = F.softmax(out, dim=1)
            score_map = score_map.cpu().data.numpy()
            score_map = score_map[0, 1:2, :, :, :]

            prediction = score_map.copy()
            prediction[prediction > 0.5] = 1
            prediction[prediction != 1] = 0

            image = image.cpu().numpy()
            label = label.cpu().numpy()

            if np.sum(prediction)==0:
                single_metric = (0,0,0,0)
                print('prediction == 0')
            else:
                single_metric = calculate_metric_percase(prediction, label)
                metric_dict['name'].append(case_name)
                metric_dict['dice'].append(single_metric[0])
                metric_dict['jaccard'].append(single_metric[1])
                metric_dict['asd'].append(single_metric[2])
                metric_dict['95hd'].append(single_metric[3])
                # print(metric_dict)

            total_metric += np.asarray(single_metric)

            if save_result:
                test_save_path_temp = os.path.join(args.test_save_path, case_name[0])
                if not os.path.exists(test_save_path_temp):
                    os.makedirs(test_save_path_temp)
                
                prediction = transpose(prediction[0].astype(np.float32))
                score_map = transpose(score_map[0].astype(np.float32))
                #print('image', image.shape)
                #print('label', label.shape)
                image = transpose(image[0, 0, ...].astype(np.float32))
                label = transpose(label[0].astype(np.float32))
                nib.save(nib.Nifti1Image(prediction, np.eye(4)), test_save_path_temp + '/'  + "pred.nii.gz")
                nib.save(nib.Nifti1Image(score_map, np.eye(4)), test_save_path_temp + '/'  + "prob.nii.gz")
                nib.save(nib.Nifti1Image(image, np.eye(4)), test_save_path_temp + '/' +  "img.nii.gz")
                nib.save(nib.Nifti1Image(label, np.eye(4)), test_save_path_temp + '/' + "gt.nii.gz")
        avg_metric = total_metric / len(testloader)
        print('average metric is {}'.format(avg_metric))
        return avg_metric

    
def main():
    args = get_args()
    if not os.path.exists(args.test_save_path):
        os.makedirs(args.test_save_path)

    transform = transforms.Compose([ToTensor()])
    db_test = ABUS(base_dir=args.root_path, fold=args.fold, split='test', transform=transform)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False,  num_workers=1, pin_memory=True)
    args.testloader = testloader

    net = VNet(n_channels=1, n_classes=args.num_classes, normalization='batchnorm', has_dropout=False, use_tm=args.use_tm).cuda()
    save_mode_path = os.path.join(args.snapshot_path, 'iter_' + str(args.start_epoch) + '.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    test(args, net, testloader, num_classes=args.num_classes, save_result=True, test_save_path=args.test_save_path)

if __name__ == '__main__':
    main()
