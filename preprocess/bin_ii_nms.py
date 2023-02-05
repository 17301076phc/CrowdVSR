'''
This script can create a sparse index_psnr list (ascending) by by throwing darts.
index is the (u,v)th patch (u - y axis, v - x axis)
low psnr indicates difficult sample, which is more worth training
'''
import json
import os
import random
from pathlib import Path

import imageio
import torch
from torch.nn.functional import interpolate
from preprocess.create_bin_ii import cal_dp_map, box_filter, cal_psnr_map

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import time
import pickle
import numpy as np
from option import args


def is_bin_file(filename, patch_size):
    return any(filename.endswith(ext) for ext in ['_ii_map_p{}.pt'.format(patch_size)])

def reverse_ii_map(psnr_map_file):
    # with open(psnr_map_file, 'rb') as _f:
    #     heatmap = pickle.load(_f)
    # reverse the map, the smaller value, the bigger after
    heatmap = 1/psnr_map_file

    return heatmap

def gen_nms_mask(patch_size_lr, threshold):
    # generate nms mask
    nms_mask = np.ones((patch_size_lr*2+1, patch_size_lr*2+1))
    center = (patch_size_lr,patch_size_lr)
    for iy in range(nms_mask.shape[0]):
        for ix in range(nms_mask.shape[1]):
            I = np.abs((iy - center[0]) * (ix - center[1])) # intersection
            U = 2*np.power(patch_size_lr,2) - I       # Union
            iou = I/U
            # print(iou)
            if iou < threshold:
                nms_mask[iy][ix] = 0
    return nms_mask

def gen_psnr_nms(heatmap, n_nms,patch_size_lr,nms_mask):
    selected_list = []
    for i in range(n_nms):
        selected = np.unravel_index(heatmap.argmax(), heatmap.shape)
        x1 = max(0, selected[1]-patch_size_lr)
        x2 = min(heatmap.shape[1], selected[1] + patch_size_lr+1)
        y1 = max(0, selected[0]-patch_size_lr)
        y2 = min(heatmap.shape[0], selected[0] + patch_size_lr+1)
        # print(x1,x2,y1,y2)
        # print(heatmap[y1:y2,x1:x2].shape)
        x3 = max(0, patch_size_lr - selected[1])
        x4 = min(patch_size_lr*2+1, heatmap.shape[1]-selected[1]+patch_size_lr)
        y3 = max(0, patch_size_lr - selected[0])
        y4 = min(patch_size_lr*2+1, heatmap.shape[0]-selected[0]+patch_size_lr)
        # print(x3,x4,y3,y4)
        # print(nms_mask[y3:y4,x3:x4].shape)
        heatmap[y1:y2,x1:x2] = nms_mask[y3:y4,x3:x4]*heatmap[y1:y2,x1:x2]

        selected_list.append(selected)
        # break
    selected_list = np.array(selected_list)
    return selected_list

def create_patches(lr,hr):
    ################## settings
    scale = 4
    patch_size_hr = 256
    patch_size_lr = patch_size_hr // scale
    threshold = 0.001
    eps = 1e-9
    n_nms = 1000

    # generate nms mask
    nms_mask = gen_nms_mask(patch_size_lr, threshold)

    lr_tensor = torch.from_numpy(lr).float()


    # get sr
    sr_tensor = interpolate(
        lr_tensor.permute(2, 0, 1).unsqueeze(0),  # (1,3,W,H)
        scale_factor=scale,
        mode='bilinear',
        align_corners=False).clamp(min=0, max=255)

    sr_tensor = sr_tensor.squeeze().permute(1, 2, 0)  # (W,H,3)
    sr = sr_tensor.numpy()

    # make shaves
    shave = scale + 6
    patch_size = patch_size_hr - shave * 2

    # precompute diff-power map
    diff_norm = (sr - hr) / 255
    diff_norm = diff_norm[shave:-shave, shave:-shave, ...]
    dpm = cal_dp_map(diff_norm, patch_size)

    # box filtering
    sum_map = box_filter(dpm, patch_size)

    # calculate psnr map
    psnr_map = cal_psnr_map(sum_map, scale, eps)

    # load ii map and reverse it
    ii_map = reverse_ii_map(psnr_map)

    # generate index_psnr list sparsed by nms
    selected_list = gen_psnr_nms(ii_map, n_nms,patch_size_lr,nms_mask)
    nums_patches = int(selected_list.shape[0]*0.1)
    patches = []
    for i in range(10):
        # i = random.randrange(0, nums_patches)
        refx, refy = int(selected_list[i][0]), int(selected_list[i][1])

        ref_patch = lr[refx: refx + 64, refy:refy + 64]
        patches.append(ref_patch)

    return patches