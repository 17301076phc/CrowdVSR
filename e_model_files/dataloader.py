import os
import random
from importlib import import_module
import cv2
import numpy as np
import torch
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
import torch.utils.data as data

# This is a simple wrapper function for ConcatDataset
import common


class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        # self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)


class Divdata(data.Dataset):
    def __init__(self,dir,scale,istrain = True):
        path1 = os.listdir(dir)
        self.scale = scale
        self.patchsize = 96
        self.dir = dir
        self.len = len(path1)
        self.istrain = istrain
        if istrain:
            n_patches = self.patchsize * 1000
            n_images = self.len
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    def __getitem__(self, idx):
        if self.istrain:
            item= idx % self.len
        else:
            item=idx
        # if self.train:
        filename = self.dir+ "/"+ f'{str(item).zfill(4)}.png'
        try:
            tmphr = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)
            height, width, _ = tmphr.shape
            hr = cv2.resize(tmphr, (width//self.scale*self.scale, height//self.scale*self.scale), interpolation=cv2.INTER_CUBIC)
            h, w, _ = hr.shape
            lr = cv2.resize(hr, (w//self.scale, h//self.scale), interpolation=cv2.INTER_CUBIC)
            # print(hr.shape,lr.shape)
            pair = self.get_patch(lr, hr)
            pair = common.set_channel(*pair, n_channels=3)
            pair_t = common.np2Tensor(*pair, rgb_range=255)

            return pair_t[0], pair_t[1]
        except:
            print(filename)


    def get_patch(self, lr, hr):

        # if self.train:
        lr, hr = common.get_patch(
            lr, hr,
            patch_size=self.patchsize, # scale2  patchsize 64
            scale=self.scale,
            input_large=False
        )

        return lr, hr

    def __len__(self):
        if self.istrain:
            return self.len*self.repeat
        return self.len
        # else:
        #     self.idx_scale = random.randint(0, len(self.scale) - 1)


class Data:
    def __init__(self):
        self.loader_train = None
        div = Divdata("div2k_dataset/div2k",scale=2)
        # sp = Divdata("SpongeBob_data")

        self.div_loader = dataloader.DataLoader(
            div,
            batch_size=64,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )
        # self.sp_loader = dataloader.DataLoader(
        #     sp,
        #     batch_size=64,
        #     shuffle=True,
        #     pin_memory=True,
        #     num_workers=4,
        # )
        # self.loader_train = dataloader.DataLoader(
        #     ConcatDataset(datasets),
        #     batch_size=64,
        #     shuffle=True,
        #     pin_memory=True,
        #     num_workers=4,
        # )

        # self.train_data_len = ConcatDataset(datasets).__len__()

        # testset = Mydata("ESPCN/data/test_hr",istrain=False)
        #
        # self.loader_test= dataloader.DataLoader(
        #         testset,
        #         batch_size=1,
        #         shuffle=False,
        #         pin_memory=True,
        #         num_workers=4,
        #     )
