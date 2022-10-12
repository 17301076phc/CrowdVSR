import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import torchvision.transforms as transforms
import skimage.color as sc


class OnlineDataset(Dataset):
    def __init__(self, max_size=2048):
        self.max_size = max_size

        self.data = []
        self.transform = transforms.Compose([transforms.ToTensor(), ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        # print(index)
        # if index<len(self.data):
        #     return self.data[index]
        # index = random.randrange(0, len(self.data))
        try:
            # print("data length is,",self.__len__())
            return self.data[index]
        except:
            print("error index,",index)

    def empty(self) -> bool:
        return len(self) == 0

    def size(self) -> int:
        return len(self)

    def augment(self,lr,hr,hor):

        # if hor:
        #     lr = lr.transpose(1,0,2)
        #     hr = hr.transpose(1,0,2)
        # else:# 垂直翻转
        lr = lr[::-1]
        hr = hr[::-1]
        lr = np2Tensor(lr)
        hr = np2Tensor(hr)
        return lr,hr

    def put(self, lr, hr):
        if len(self) >= self.max_size:
            # self.data = self.data[self.max_size // 4:]
            self.data = self.data[60:]

        # lr = self.transform(lr)
        # hr = self.transform(hr)
        # a_lr,a_hr = self.augment(lr,hr,True)
        # self.data.append((a_lr, a_hr))
        lr = np2Tensor(lr)
        hr = np2Tensor(hr)
        self.data.append((lr, hr))



def np2Tensor(img, rgb_range=255):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor