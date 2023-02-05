import math
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy import dot, average
from scipy import linalg
from torch import nn
from torchvision.transforms import Compose, ToTensor,CenterCrop

from model import MultiNetwork
from new_model import OurNetwork

network_config = {4: {'block': 8, 'feature': 48},
                  3: {'block': 8, 'feature': 42},
                  2: {'block': 8, 'feature': 26},
                  1: {'block': 1, 'feature': 26}}

# our_network_config = {4: {'block': 12, 'feature': 48},
#                   3: {'block': 8, 'feature': 42},
#                   2: {'block': 8, 'feature': 26},
#                   1: {'block': 1, 'feature': 26}}



def load_model():
    # model = MultiNetwork(network_config)
    model = OurNetwork(network_config)

    model = model.to(torch.device('cuda'))

    # model_path = 'model/div2k_x3.pth'
    # model_path = 'model/div2k_x2.pth'
    # model_path = 'model/div2k_x4.pth'

    # model_path = 'our_model/naf_x2.pth'
    model_path = 'our_model/naf_x3.pth'
    # model_path = 'our_model/naf_x4.pth'

    if not Path(model_path).exists():
        print('model not exists')
        exit(-1)

    model.load_state_dict(torch.load(model_path))
    # print(model.state_dict().keys())
    # for name, value in model.named_parameters():
    #     # print(name)
    #     if name == 'networks.2.head':
    #         value.requires_grad = False

    return model


def load_e_model():
    from e_model_files.e_model import ENetwork
    # model = ENetwork("eSR-TR_s2_K3_C4")
    # model = ENetwork("eSR-TR_s2_K3_C8")
    # model = ENetwork("eSR-TR_s2_K3_C1")
    model = ENetwork("eSR-TR_s2_K3_C3")


    model = model.to(torch.device('cuda'))
    # model_path = 'e_model_files/e_model/eSR-TR_s2_K3_C16.pth'
    # model_path = 'e_model_files/e_model/eSR-TR_s2_K3_C8.pth'
    # model_path = 'e_model_files/e_model/eSR-TR_s2_K3_C4.pth'
    # model_path = 'e_model_files/e_model/eSR-TR_s2_K3_C1.pth'
    model_path = 'e_model_files/e_model/eSR-TR_s2_K3_C3.pth'


    if not Path(model_path).exists():
        print('model not exists')
        exit(-1)

    model.load_state_dict(torch.load(model_path), strict=False)
    # print(model.state_dict().keys())
    return model


def train_one_epoch(model, loader, scale):
    model.set_target_scale(scale)

    model.train()
    loss_fn = nn.L1Loss()
    # loss_fn = nn.MSELoss()
    # x2  lr=1e-7   x3 lr=1e-6
    optimizer = torch.optim.Adam(model.networks[scale - 1].parameters(), lr=1e-6, weight_decay=0)
    avg_loss = []
    for iteration, data in enumerate(loader, 1):
        input_tensor, target_tensor = data[0].cuda(), data[1].cuda()
        optimizer.zero_grad()
        output_tensor = model(input_tensor)
        loss = loss_fn(output_tensor, target_tensor)
        avg_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    # print("loss is ",sum(avg_loss)/len(avg_loss))

def train_one_epoch_emodel(model, loader, scale):

    model.train()
    loss_fn = nn.L1Loss()
    # loss_fn = nn.SmoothL1Loss()
    # loss_fn = nn.MSELoss()
    # x2  lr=1e-7   x3 lr=1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-8, weight_decay=0)
    avg_loss = []
    for iteration, data in enumerate(loader, 1):
        input_tensor, target_tensor = data[0].cuda(), data[1].cuda()
        optimizer.zero_grad()
        output_tensor = model(input_tensor)
        loss = loss_fn(output_tensor, target_tensor)
        avg_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    # print("loss is ",sum(avg_loss)/len(avg_loss))

# 推断结果
# def inference(model, frame, scale):
#     model.set_target_scale(scale)
#     model.eval()
#     device = torch.device('cuda')
#     transform = Compose([ToTensor(), ])
#     with torch.no_grad():
#         # input_tensor_ = torch.from_numpy(frame).cuda()
#         # input_tensor_ = input_tensor_.permute(2, 0, 1)
#         # input_tensor_ = input_tensor_.true_divide(255)
#         input_tensor_ = transform(frame).to(device)
#         input_tensor_.unsqueeze_(0)
#
#         output_ = model(input_tensor_)
#         output_ = output_.data[0].permute(1, 2, 0)
#         output_ = output_ * 255
#         output_ = torch.clamp(output_, 0, 255)
#         # sr = quantize(output_).squeeze(0)
#         # normalized = sr * 255 / 255
#         # ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
#     torch.cuda.synchronize()
#     return output_.cpu().numpy().astype(np.uint8)

def inference(model, frame, scale,is_emodel):
    if not is_emodel:
        model.set_target_scale(scale)
    model.eval()

    with torch.no_grad():
        input_tensor_ = torch.from_numpy(frame).byte().cuda()
        input_tensor_ = input_tensor_.permute(2, 0, 1)
        input_tensor_ = input_tensor_.true_divide(255)
        input_tensor_.unsqueeze_(0)

        output_ = model(input_tensor_)
        output_ = output_.data[0].permute(1, 2, 0)
        output_ = output_ * 255
        output_ = torch.clamp(output_, 0, 255)

    torch.cuda.synchronize()
    return output_.cpu().numpy().astype(np.uint8)

def quantize(img, rgb_range=255):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(hr, sr,scale):
    diff = (sr/1.0 - hr/1.0) / 255.0
    # print(diff)
    shave = scale + 6

    valid = diff[shave:-shave, shave:-shave, ...]
    mse = np.mean(np.power(valid,2))
    # print(mse)

    return -10 * math.log10(mse)
    # return 10. * math.log10(255.0**2 / np.mean((hr/1.0 - sr/1.0) ** 2 ))


def np2Tensor(img, rgb_range=255):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor

def get_patch(lr,hr,patchsize,scale):
    patch_size = 64
    scale =4
    patches = []
    height, width, _ = lr.shape
    m, n = width // patch_size, height // patch_size
    for i in range(m):
        for j in range(n):
            roi = lr[j * patch_size: (j + 1) * patch_size, i * patch_size:(i + 1) * patch_size]
            hr_roi = hr[j * patch_size * scale:(j + 1) * patch_size * scale,
                     i * patch_size * scale:(i + 1) * patch_size * scale]
            patches.append((roi,hr_roi))
    return patches

def sim(image1, image2):
    def pHash(img, leng=32, wid=32):
        img = cv2.resize(img, (leng, wid))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dct = cv2.dct(np.float32(gray))
        dct_roi = dct[0:8, 0:8]
        avreage = np.mean(dct_roi)
        phash_01 = (dct_roi > avreage) + 0
        phash_list = phash_01.reshape(1, -1)[0].tolist()
        hash = ''.join([str(x) for x in phash_list])
        return hash

    def Hamming_distance(hash1, hash2):
        num = 0
        for index in range(len(hash1)):
            if hash1[index] != hash2[index]:
                num += 1
        return num
    hash1 = pHash(image1)
    hash2 = pHash(image2)
    return Hamming_distance(hash1,hash2)