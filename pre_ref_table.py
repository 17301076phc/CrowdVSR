import json
import queue
from pathlib import Path

import numpy as np

from preprocess import bin_ii_darts, bin_ii_single
import cv2

class PatchItem:
    def __init__(self,  xy,filename,distance: int):
        # self.lr = lr
        # self.hr = hr
        self.xy =xy
        self.filename = filename
        self.distance = distance

    def __lt__(self, other):
        return self.distance < other.distance

cap1 = cv2.VideoCapture()
cap1.open("video/10lr.mp4")
cap2 = cv2.VideoCapture()
cap2.open("video/10.mp4")
sample_interval = 10
patch_size = 64
scale = 4
candidates = list(Path("candidates").iterdir())
def make_table(pts):
    print("creating patch-hash table .......  .........")
    table = []
    for candidate in candidates:
        l = len(list(Path(candidate).iterdir()))
        for i in range(max(0, pts - 30), pts+30):
            filename = candidate / f'{str(i).zfill(4)}.png'

            hr = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)
            # lr = cv2.cvtColor(cv2.imread(str(lr_filename)), cv2.COLOR_BGR2RGB)
            lr = cv2.resize(hr, (480, 270), interpolation=cv2.INTER_CUBIC)
            patch_hash_table = json.loads(
                open("patch_hash/" + str(candidate)[11:] + f'/{str(i).zfill(4)}.txt').read())
            # print(len(patch_hash_table))
            for item in range(1000):
                xy = patch_hash_table[item]['xy']

                lr_i = lr[xy[0]: xy[0] + patch_size, xy[1]:xy[1] + patch_size]
                # hr_i = hr[xy[0] * scale:xy[0] * scale + patch_size * scale,
                #        xy[1] * scale:xy[1] * scale + patch_size * scale]
                table.append((lr_i, xy,filename))

    print("finish creating table ...")
print("start creating patches......")

result_file = Path(f'pre_select_patch.txt').open('w', encoding='utf8')
count = 0
while True:
    ret, lr_frame = cap1.read()
    if not ret: break
    lr_frame = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)
    retf, hr_frame = cap2.read()
    hr_frame = cv2.cvtColor(hr_frame, cv2.COLOR_BGR2RGB)
    for _ in range(sample_interval - 1):
        _, _ = cap1.read()
        _, _ = cap2.read()
    ref_patches = bin_ii_single.create(lr_frame,hr_frame)
    count +=1
    result =[]
    table = make_table(count)
    patch_queue = queue.PriorityQueue()
    for ref_patch in ref_patches:
        reference_hash = cv2.img_hash.pHash(ref_patch)[0]
        for lr_i,xy,filename in table:
            # hr = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)
            # lr = cv2.resize(hr, (480, 270), interpolation=cv2.INTER_CUBIC)
            # lr_i = lr[xy[0]: xy[0] + patch_size, xy[1]:xy[1] + patch_size]
            p = cv2.img_hash.pHash(lr_i)[0]
            distance = sum([bin(x ^ y).count('1') for x, y in zip(reference_hash, p)])
            # print(distance)
            patch_queue.put(PatchItem(xy,filename, distance))

    for i in range(100):
        item = patch_queue.get()
        tmp = {
            'xy':item.xy,
            'filename': str(item.filename)
        }
        result.append(tmp)
    print("once ")
    result_file.write(json.dumps(result))
    result_file.flush()