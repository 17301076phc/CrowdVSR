import glob
import os
import pickle
from pathlib import Path

import cv2
import imageio


def _check_and_load( img, f):
    print('Making a binary: {}'.format(f))
    with open(f, 'wb') as _f:
        print(imageio.imread(img).shape)
        pickle.dump(imageio.imread(img), _f)

# def _scan():
#     for candidate in candidates:
#         # l = len( list(Path(candidate).iterdir()))
#         names_hr = sorted(
#             glob.glob(os.path.join(candidate, '*' + ext[0]))
#         )

def creatlr():
    apath = "online_data"
    lrpath = "online_data_lr_x4"
    l = len( list(Path(apath).iterdir()))
    print(l)
    for i in range(l):
        filename = apath + f'/{str(i).zfill(4)}.png'
        # print(str(filename)[11:])
        try:
            hr = cv2.imread(str(filename))
            height, width, _ = hr.shape
            if height<48*4 or width<48*4:
            # if height<128 or width<128:
                print(filename)
            lr = cv2.resize(hr, (width // scale, height // scale), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(lrpath + f'/{str(i).zfill(4)}.png', lr)
        except:
            print()
            print(str(filename))
            break


print("finish lr creating.......")

if __name__ == '__main__':
    # args.ext = "sep+reset"
    scale = 4
    # apath = "candidates"
    # apath = "online_data"
    # candidates = list(Path(apath).iterdir())
    # ext = ('.png', '.png')
    # # path_bin = "candidates_bin"
    # # lrpath = "candidates_lr_x3/"
    # lrpath = "online_data_lr_x3"
    # lrcandidates = list(Path(lrpath).iterdir())

    print("creating lr.......")
    # for candidate in candidates:
    #
    #     # if str(candidate)=='candidates/broadcaster1':
    #         print(candidate)
    #         l = len( list(Path(candidate).iterdir()))
    #         # l = len( list(Path(apath).iterdir()))
    #         print(l)
    #         for i in range(l):
    #             filename = candidate / f'{str(i).zfill(4)}.png'
    #             # print(str(filename)[11:])
    #             try:
    #                 hr = cv2.imread(str(filename))
    #                 height, width, _ = hr.shape
    #                 lr = cv2.resize(hr, (width//scale, height//scale), interpolation=cv2.INTER_CUBIC)
    #                 cv2.imwrite(lrpath+str(filename)[11:],lr)
    #             except:
    #                 print()
    #                 print(str(filename))
    #                 break
    creatlr()
    print("finish lr creating.......")

    # print("creating bin.......")
    # for candidate in candidates:
    #     # l = len( list(Path(candidate).iterdir()))
    #     names_hr = sorted(
    #         glob.glob(os.path.join(candidate, '*' + ext[0]))
    #     )
    #     for h in names_hr:
    #         b = h.replace(apath, path_bin)
    #         b = b.replace(ext[0], '.pt')
    #         _check_and_load(h, b)
    # print("creating lr bin............")
    # for candidate in lrcandidates:
    #     # l = len( list(Path(candidate).iterdir()))
    #     names_lr = sorted(
    #         glob.glob(os.path.join(candidate, '*' + ext[0]))
    #     )
    #     for h in names_lr:
    #         b = h.replace("candidates_lr", "candidates_bin_lr")
    #         b = b.replace(ext[0], '.pt')
    #         _check_and_load(h, b)
    # print("finish bin creating........")


    # f = "candidates_lr/broadcaster1/0000.png"
    # h = "candidates/broadcaster1/0000.png"
    #
    # hr = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
    # print( cv2.img_hash.pHash(cv2.imread(f)))

    # print( cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)==imageio.imread(f))
    # lr = cv2.resize(hr, (480, 270), interpolation=cv2.INTER_CUBIC)
    # print(lr.shape)
    # b = f.replace('candidates_lr/broadcaster1/', '')
    # b = b.replace('.png', '.pt')
    # # images_lr[i].append(b)
    # _check_and_load(f, b)
    # with open(b, 'rb') as _f:
    #     lr = pickle.load(_f) # (W,H,3)
    #     print(lr==cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))
