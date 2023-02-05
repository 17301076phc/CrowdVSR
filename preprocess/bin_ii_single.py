'''
Once bin files of DIV2K dataset has been generated
This script can create an index list sorted by psnr (ascending)
index is the (u,v)th patch (u - y axis, v - x axis)
low psnr indicates difficult sample, which is more worth training
'''
import json
import os
import pickle
import random
from pathlib import Path

import cv2
import imageio
from PIL import Image

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
# import pickle
import torch
import time
from torch.nn.functional import interpolate
from numba import jit


# cumsum in Numba currently only supports the first argument. I.e. none of axis, dtype or out are implemented.
def box_filter(imSrc, patch_size):
    '''BOXFILTER   O(1) time box filtering using cumulative sum.

    Definition imDst(x, y)=sum(sum(imSrc(x:x+r,y:y+r))).
    Running time independent of r.

    Args:
        imSrc (np.array): source image, shape(hei,wid).
        patch_size (int): box filter size. (r)

    Returns:
        imDst (np.array): img after filtering, shape(hei-r+1,wid-r+1).
    '''
    [hei, wid] = imSrc.shape
    imDst = np.zeros_like(imSrc)

    # cumulative sum over Y axis
    imCum = np.cumsum(imSrc, axis=0)
    imDst[0, :] = imCum[patch_size - 1, :]
    imDst[1:hei - patch_size + 1, :] = imCum[patch_size:, :] - imCum[0:hei - patch_size, :]

    # cumulative sum over X axis
    imCum = np.cumsum(imDst, axis=1)
    imDst[:, 0] = imCum[:, patch_size - 1]
    imDst[:, 1:wid - patch_size + 1] = imCum[:, patch_size:] - imCum[:, 0:wid - patch_size]

    # cut the desired area
    imDst = imDst[:hei - patch_size + 1, :wid - patch_size + 1]

    return imDst


@jit(nopython=True)
def cal_dp_map(diff_norm, patch_size):
    diff_norm_pow = np.power(diff_norm, 2)
    dpm = np.sum(diff_norm_pow, axis=2)
    mn = patch_size * patch_size
    dpm = dpm / (mn * 3)  # channel = 3
    return dpm


@jit(nopython=True)
def cal_psnr_map(sum_map, scale, eps):
    sum_map = sum_map[::scale, ::scale]
    sum_map = sum_map + eps  # avoid zero value
    psnr_map = -10 * np.log10(sum_map)
    return psnr_map


@jit(nopython=True)
def psnr_sort(psnr_map, iy, ix):
    index_psnr = np.hstack((iy, ix, psnr_map.reshape(-1, 1)))  # 水平组合
    sort_index = np.argsort(index_psnr[:, -1])
    index_psnr = index_psnr[sort_index]
    return index_psnr


def create_patches(lrimg,hrimg):
    ################## settings
    eps = 1e-9
    rgb_range = 255
    scale = 2
    hr_patch_size = 128

    # hr_patch_size = 256  # the size is for hr patch

    lr_tensor = torch.from_numpy(lrimg).float()

    # get sr
    sr_tensor = interpolate(
        lr_tensor.permute(2, 0, 1).unsqueeze(0),  # (1,3,W,H)
        scale_factor=scale,
        mode='bilinear',
        # mode='bicubic',
        align_corners=False).clamp(min=0, max=255)

    sr_tensor = sr_tensor.squeeze().permute(1, 2, 0)  # (W,H,3)
    sr = sr_tensor.numpy()

    # make shaves
    shave = scale + 6
    patch_size = hr_patch_size - shave * 2

    # precompute diff-power map
    diff_norm = (sr - hrimg) / rgb_range
    diff_norm = diff_norm[shave:-shave, shave:-shave, ...]
    dpm = cal_dp_map(diff_norm, patch_size)

    # box filtering
    sum_map = box_filter(dpm, patch_size)

    # calculate psnr map
    psnr_map = cal_psnr_map(sum_map, scale, eps)
    [hei, wid] = psnr_map.shape

    # generate index
    iy = np.arange(hei).reshape(-1, 1).repeat(wid, axis=1).reshape(-1, 1)
    ix = np.arange(wid).reshape(1, -1).repeat(hei, axis=0).reshape(-1, 1)
    # print(iy,ix)

    # sort index by psnr
    index_psnr = psnr_sort(psnr_map, iy, ix)
    # print(index_psnr[100])
    # print(index_psnr)
    nums_patches = int(index_psnr.shape[0]*0.01)
    print("reference patch number",nums_patches)
    patches = []
    for i in range(nums_patches):
        # i = random.randrange(0, nums_patches)
        refx, refy = int(index_psnr[i][0]), int(index_psnr[i][1])
        # print(refy,refx)

        ref_patch = lrimg[refx: refx + 64, refy:refy + 64]
        patches.append(ref_patch)
        # Image.fromarray(ref_patch).save("selector_patches/"+str(i)+".png")

    return patches
