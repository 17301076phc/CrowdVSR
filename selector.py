import json
import queue
import random
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from numba import jit

from preprocess import bin_ii_single, bin_ii_darts,bin_ii_nms
from process import sim
import imagehash
from option import args

class PatchItem:
    def __init__(self, lr: np.ndarray, hr: np.ndarray, distance: int):
        self.lr = lr
        self.hr = hr
        self.distance = distance

    def __lt__(self, other):
        return self.distance < other.distance
        # return self.distance > 0.75


class Selector:
    def __init__(self):
        self.sample_rate = 1
        self.scale = 3
        self.patch_size = 64 #64

        self.candidates = list(Path(args.candidates).iterdir())
        self.candidate_pts_range = 3
        self.reference_patch_length = 8
        self.candidate_patch_length = 128

    def _select_reference_patch(self, reference_frame: List[np.ndarray]) -> List[np.ndarray]:
        patch_size = self.patch_size
        current_frame = reference_frame.pop()
        height, width, _ = current_frame.shape

        m, n = width // patch_size, height // patch_size
        mse_table = np.zeros(m * n)
        for i in range(m):
            for j in range(n):
                current_roi = current_frame[j * patch_size: (j + 1) * patch_size, i * patch_size:(i + 1) * patch_size]
                for frame in reference_frame:
                    reference_roi = frame[j * patch_size: (j + 1) * patch_size, i * patch_size:(i + 1) * patch_size]
                    mse = np.mean((current_roi - reference_roi) ** 2)
                    mse_table[i * n + j] += mse  # 一维数组形式保存

        reference_patches = []
        indexes = mse_table.argsort()[::-1][:self.reference_patch_length]

        for idx in indexes:
            i, j = idx // n, idx % n
            x = current_frame[j * patch_size: (j + 1) * patch_size, i * patch_size:(i + 1) * patch_size]
            reference_patches.append(x)
            # y = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
            # cv2.imwrite("selector_patches/"+str(i)+".png",y)

        return reference_patches

    def select_patches(self, reference_frame: List[np.ndarray], pts: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        candidates = []
        patch_size = self.patch_size
        scale = self.scale
        reference_patches = self._select_reference_patch(reference_frame)
        reference_patches = [cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in reference_patches]
        # print(self.candidates)
        try:
            for candidate in self.candidates:
                for i in range(max(0, pts - self.candidate_pts_range), pts):
                # l = len( list(Path(candidate).iterdir()))
                # for i in range(0,l,100):
                    filename = candidate / f'{str(i).zfill(4)}.png'
                    # print(filename)
                    hr = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)
                    # hr = cv2.imread(str(filename))
                    height, width, _ = hr.shape
                    height, width = height // scale, width // scale
                    lr = cv2.resize(hr, (width, height), interpolation=cv2.INTER_CUBIC)
                    candidates.append((lr, hr))
        except:
            print(filename)
        # print(len(self.candidates))
        print(len(candidates)) #15
        print(len(reference_patches))
        patch_queue = queue.PriorityQueue()
        for ref_patch in reference_patches:
            reference_hash = cv2.img_hash.pHash(ref_patch)[0]
            # ref_hash = imagehash.dhash(Image.fromarray(ref_patch))
            for lr, hr in candidates:
                height, width, _ = lr.shape
                m, n = width // patch_size, height // patch_size
                for i in range(m):
                    for j in range(n):
                        roi = lr[j * patch_size: (j + 1) * patch_size, i * patch_size:(i + 1) * patch_size]
                        hr_roi = hr[j * patch_size * scale:(j + 1) * patch_size * scale,
                                 i * patch_size * scale:(i + 1) * patch_size * scale]

                        roi_hash = cv2.img_hash.pHash(roi)[0]
                        distance = sum([bin(x ^ y).count('1') for x, y in zip(reference_hash, roi_hash)])

                        # roi_hash = imagehash.dhash(Image.fromarray(roi))
                        # distance = ref_hash-roi_hash
                        # distance = his_calculate(ref_patch, roi)
                        # print(distance)
                        patch_queue.put(PatchItem(roi, hr_roi, distance))

        patches = []
        for i in range(self.candidate_patch_length):
            item = patch_queue.get()
            # print(item.distance)
            patches.append((item.lr, item.hr))
            # cv2.imwrite("simliar_patches/hr/"+str(i)+".png", item.hr)
            # cv2.imwrite("simliar_patches/lr/"+str(i)+".png", item.lr)
        return patches


class RandomSelector:
    def __init__(self):
        self.candidates = list(Path(args.candidates).iterdir())
        self.patch_size = 64 # 64
        self.scale = 3

    def select_patch(self) -> (np.ndarray, np.ndarray):
        candidate = self.candidates[random.randrange(0, len(self.candidates))]
        candidate_size = len(list(candidate.iterdir()))
        frame_id = random.randrange(0, candidate_size)#  high resolution frames
        # print(str(candidate / f'{str(frame_id).zfill(4)}.png'))
        frame = cv2.imread(str(candidate / f'{str(frame_id).zfill(4)}.png'))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # if not frame:
        #     print(str(candidate / f'{str(frame_id).zfill(4)}.png'))
        scale = self.scale
        patch_size = self.patch_size

        height, width, _ = frame.shape
        height, width = height // scale, width // scale
        lr = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        x = random.randrange(0, width - patch_size + 1)
        y = random.randrange(0, height - patch_size + 1)
        lr_patch = lr[y:y + patch_size, x:x + patch_size]
        hr_patch = frame[y * scale:(y + patch_size) * scale, x * scale:(x + patch_size) * scale]

        return lr_patch, hr_patch

    def select_patches(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        patches = []
        for _ in range(10):
            candidate = self.candidates[random.randrange(0, len(self.candidates))]
            candidate_size = len(list(candidate.iterdir()))
            frame_id = random.randrange(0, candidate_size)
            frame = cv2.imread(str(candidate / f'{str(frame_id).zfill(4)}.png'))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            scale = self.scale
            patch_size = self.patch_size

            height, width, _ = frame.shape
            height, width = height // scale, width // scale
            lr = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

            for _ in range(64):
                x = random.randrange(0, width - patch_size + 1)
                y = random.randrange(0, height - patch_size + 1)
                lr_patch = lr[y:y + patch_size, x:x + patch_size]
                hr_patch = frame[y * scale:(y + patch_size) * scale, x * scale:(x + patch_size) * scale]
                patches.append((lr_patch, hr_patch))

        return patches

class CandidatesSelector:
    def __init__(self):
        self.candidates = list(Path(args.candidates).iterdir())
        self.patch_size = 64 # 64
        self.scale = 2
        self.candidate_patch_length = 128
        self.reference_patch_length = 10
        self.candidate_pts_range = 30
        # self.patch_hash = "patch_hash"
        self.repeat = True

    def _select_reference_patch(self, reference_frame,hr) -> List[np.ndarray]:
        # patches = []
        # if len(reference_frame) == len(hr):
        #     for i in range(len(reference_frame)):
        #         # patches += bin_ii_darts.create(reference_frame[i], hr[i])
        #         patches += bin_ii_single.create_patches(reference_frame[i], hr[i])
        #         # patches += bin_ii_nms.create_patches(reference_frame[i], hr[i])
        # else:
        #     print("not equal!!")

        patch_size = self.patch_size
        reference_patches = []

        while reference_frame:
            current_frame = reference_frame.pop()
            height, width, _ = current_frame.shape

            m, n = width // patch_size, height // patch_size
            mse_table = np.zeros(m * n)
            for i in range(m):
                for j in range(n):
                    current_roi = current_frame[j * patch_size: (j + 1) * patch_size, i * patch_size:(i + 1) * patch_size]
                    for frame in reference_frame:
                        reference_roi = frame[j * patch_size: (j + 1) * patch_size, i * patch_size:(i + 1) * patch_size]
                        mse = np.mean((current_roi - reference_roi) ** 2)
                        mse_table[i * n + j] += mse  # 一维数组形式保存

            indexes = mse_table.argsort()[::-1][:self.reference_patch_length]

            for idx in indexes:
                i, j = idx // n, idx % n
                x = current_frame[j * patch_size: (j + 1) * patch_size, i * patch_size:(i + 1) * patch_size]
                reference_patches.append(x)

        # patch_size = self.patch_size
        # reference_patches = []
        #
        # while reference_frame:
        #     current_frame = reference_frame.pop()
        #     height, width, _ = current_frame.shape
        #
        #     m, n = width // patch_size, height // patch_size
        #     m1, n1 = width % patch_size, height % patch_size
        #     h_size = patch_size//2
        #     m, n = width // patch_size, height // patch_size
        #     for i in range(m):
        #         for j in range(n):
        #             current_roi = current_frame[j * patch_size: (j + 1) * patch_size, i * patch_size:(i + 1) * patch_size]
        #             reference_patches.append(current_roi)
            # for i in range(h_size,width,patch_size):
            #     for j in range(h_size,height,patch_size):
            #         current_roi = current_frame[j - h_size: j + h_size, i - h_size:i + h_size]
            #         reference_patches.append(current_roi)

            # for i in range(width,h_size,-patch_size):
            #     for j in range(height,h_size,-patch_size):
            #         current_roi = current_frame[j - h_size: j + h_size, i - h_size:i + h_size]
            #         reference_patches.append(current_roi)

        print("reference_patches length:",len(reference_patches))
        return reference_patches

        # return patches

    def select_patches(self,reference_frame,pts,h) -> List[Tuple[np.ndarray, np.ndarray]]:
        patch_size = self.patch_size
        scale = self.scale
        reference_patches = self._select_reference_patch(reference_frame,h)
        # reference_patches = [cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in reference_patches]
        # for r in range(len(reference_patches)):
        #     Image.fromarray(reference_patches[r]).save("selector_patches/" + str(r) + ".png")
        start = time.time()
        table = []
        for candidate in self.candidates:
            l = len(list(Path(candidate).iterdir()))
            # for i in range(0, l,3):
            for i in range(max(0, pts - 30), min(pts+self.candidate_pts_range,l)):
                filename = candidate / f'{str(i).zfill(4)}.png'
                # lr_filename = "candidates_lr/" + str(candidate)[11:] + f'/{str(i).zfill(4)}.png'
                # print(filename)
                hr = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)
                # lr = cv2.cvtColor(cv2.imread(str(lr_filename)), cv2.COLOR_BGR2RGB)
                # img = Image.fromarray(hr.astype('uint8'))
                # lr = np.array(img.resize((480, 270), resample=Image.BICUBIC))
                lr = cv2.resize(hr, (hr.shape[1]//scale, hr.shape[0]//scale), interpolation=cv2.INTER_CUBIC)
                # patch_hash_table = json.loads(
                #     open("patch_hash/" + str(candidate)[11:] + f'/{str(i).zfill(4)}.txt').read())
                patch_hash_table = json.loads(
                    open("patch_hash_x2/" + str(candidate)[11:] + f'/{str(i).zfill(4)}.txt').read())

                # print(len(patch_hash_table))
                for i in range(50):
                    item = i
                    # item = random.randrange(0,int(len(patch_hash_table)*0.1))
                    xy = patch_hash_table[item]['xy']

                    lr_i = lr[xy[0]: xy[0] + patch_size, xy[1]:xy[1] + patch_size]
                    hr_i = hr[xy[0] * scale:xy[0] * scale + patch_size * scale,
                           xy[1] * scale:xy[1] * scale + patch_size * scale]
                    table.append((lr_i,hr_i))

        end = time.time()
        print("creating table cost: ",end-start)

        patch_queue = queue.PriorityQueue()
        for ref_patch in reference_patches:
            reference_hash = cv2.img_hash.pHash(ref_patch)[0]
            for lr,hr in table:
                # lr_t = cv2.cvtColor(lr, cv2.COLOR_RGB2BGR)
                p = cv2.img_hash.pHash(lr)[0]
                distance = sum([bin(x ^ y).count('1') for x, y in zip(reference_hash, p)])
                # distance = sim(ref_patch,lr)
                # # print(distance)
                patch_queue.put(PatchItem(lr, hr, distance))

        end2 = time.time()
        print("phash time cost: ",end2-end)

        patches = []
        for i in range(self.candidate_patch_length):
        # for i in range(60):
            item = patch_queue.get()
            if self.repeat:
                # for _ in range(2):
                    patches.append((item.lr, item.hr))
            # Image.fromarray(item.hr).save("simliar_patches/hr/" + str(i) + ".png")
            # Image.fromarray(item.lr).save("simliar_patches/lr/" + str(i) + ".png")

        return patches

class RandomSelector2:
    def __init__(self):
        self.candidates = list(Path(args.candidates).iterdir())
        self.patch_size = 64 # 64
        self.scale = 2

    def select_patch(self) -> (np.ndarray, np.ndarray):
        patches = []
        candidate = self.candidates[random.randrange(0, len(self.candidates))]
        # candidate_size = len(list(candidate.iterdir()))
        # print(candidate_size)
        frame_id = random.randrange(0, 30)#  high resolution frames
        # print(str(candidate / f'{str(frame_id).zfill(4)}.png'))

        frame = cv2.imread(str(candidate / f'{str(frame_id).zfill(4)}.png'))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        height, width = height // self.scale, width // self.scale
        lr = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

        patch_hash_table = json.loads(open("patch_hash_x2/" + str(candidate)[11:] + f'/{str(frame_id).zfill(4)}.txt').read())
        nums = int(len(patch_hash_table)*0.1)
        i = random.randrange(0,30)
        xy = patch_hash_table[i]['xy']
        lr_patch = lr[xy[0]:xy[0] + self.patch_size, xy[1]:xy[1] + self.patch_size]
        hr_patch = frame[xy[0] * self.scale:(xy[0] + self.patch_size) * self.scale,
                   xy[1] * self.scale:(xy[1] + self.patch_size) * self.scale]
        patches.append((lr_patch, hr_patch))

        return patches

    def select_patches(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        patches = []
        for candidate in self.candidates:
            for _ in range(3):
                # candidate = self.candidates[random.randrange(0, len(self.candidates))]
                # print(candidate_size)
                frame_id = random.randrange(0, 10)  # high resolution frames
                # print(str(candidate / f'{str(frame_id).zfill(4)}.png'))

                frame = cv2.imread(str(candidate / f'{str(frame_id).zfill(4)}.png'))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = frame.shape
                height, width = height // self.scale, width // self.scale
                lr = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

                patch_hash_table = json.loads(
                    open("patch_hash_x2/" + str(candidate)[11:] + f'/{str(frame_id).zfill(4)}.txt').read())
                nums = int(len(patch_hash_table) * 0.1)
                # i = random.randrange(0, 20)
                for i in range(3):
                    xy = patch_hash_table[i]['xy']
                    lr_patch = lr[xy[0]:xy[0] + self.patch_size, xy[1]:xy[1] + self.patch_size]
                    hr_patch = frame[xy[0] * self.scale:(xy[0] + self.patch_size) * self.scale,
                               xy[1] * self.scale:(xy[1] + self.patch_size) * self.scale]
                    patches.append((lr_patch, hr_patch))

        return patches
