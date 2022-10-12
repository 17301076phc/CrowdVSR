import asyncio
import io
import json
import time
from pathlib import Path

from PIL import Image
import cv2
import numpy as np
import requests
import torch
import torch.multiprocessing as mp
from aiortc import RTCPeerConnection, RTCSessionDescription
# from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from dataset import OnlineDataset
from imresize import imresize
from option import args
from process import load_model, train_one_epoch, inference,calc_psnr
from selector import Selector, RandomSelector, CandidatesSelector, RandomSelector2


async def run(frame_queue: mp.Queue, patch_queue: mp.Queue):
    pc = RTCPeerConnection()

    @pc.on('track')
    async def on_track(track):
        frame_id = 0
        while True:
            frame = await track.recv()
            frame = frame.to_ndarray(format='rgb24')
            if frame_id % args.sample_interval == 0:
                frame_queue.put(frame)
                patch_queue.put(frame)
                # cv2.imwrite("selector_patches/f.png",frame)
                # print(frame.shape[2]) #3
                # Image.fromarray(frame).save("tmplr.png")

            frame_id += 1

    @pc.on('icegatheringstatechange')
    async def on_icegatheringstatechange():
        print('icegatheringstate: ', pc.iceGatheringState)

    # negotiate
    pc.addTransceiver('video', direction='recvonly')

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # todo: icegatheringstatechange
    r = requests.post(f'http://127.0.0.1:{args.port}/offer', json={
        'sdp': pc.localDescription.sdp,
        'type': pc.localDescription.type,
    })
    answer = r.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer['sdp'], type=answer['type']))

    await asyncio.sleep(3600)

# 获取结果
def render(frame_queue: mp.Queue, state_queue: mp.Queue):
    torch.cuda.set_device(3)
    pretrain_model = load_model()
    model = load_model()
    model.cuda()
    model.eval()

    # result_file = Path(f'result_{int(time.time())}.txt').open('w', encoding='utf8')
    # result_file = Path(f'result_org_x4.txt').open('w', encoding='utf8')
    result_file = Path(f'result_x2.txt').open('w', encoding='utf8')
    # result_file = Path(f'result_org_x2.txt').open('w', encoding='utf8')
    # result_file = Path(f'result_x3.txt').open('w', encoding='utf8')
    # result_file = Path(f'result_org_x3.txt').open('w', encoding='utf8')


    cap = cv2.VideoCapture()
    cap.open(args.hr_video)
    video = cv2.VideoWriter("resultvideo.avi", cv2.VideoWriter_fourcc(*"MJPG"), 1,
                            (1920,1080))
    async def show_frame():
        pts = 0
        scale = 2
        sleeptime = 0
        while True:
            # update model
            buffer = None
            while state_queue.qsize() > 0:
                buffer = state_queue.get()
            if buffer:
                print("model load......")
                model.load_state_dict(torch.load(buffer))

            if not frame_queue.empty():
                frame = frame_queue.get()

                ret, hr_frame = cap.read()
                # cv2.imwrite("tmphr.png", hr_frame)
                hr_frame = cv2.cvtColor(hr_frame, cv2.COLOR_BGR2RGB)
                for _ in range(args.sample_interval - 1):
                    _, _ = cap.read()
                # bicubic_frame = imresize(frame.astype('uint8'),scalar_scale=2)
                img = Image.fromarray(frame.astype('uint8'))
                bicubic_frame = np.array(img.resize((1920, 1080),resample=Image.BICUBIC))
                # bicubic_frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                pretrain_frame = inference(pretrain_model, frame, scale)
                sr_frame = inference(model, frame, scale)

                # make video
                # cv2.imwrite("ttmp.png", sr_frame)
                # sframe = cv2.imread("ttmp.png")
                sframe = sr_frame[:, :, ::-1]

                # sframe = cv2.cvtColor(sr_frame, cv2.COLOR_BGR2RGB)
                print(sframe.shape[1],sframe.shape[0])
                video.write(sframe)

                avg_bicubic_frame_psnr = calc_psnr(hr_frame, bicubic_frame)
                avg_pretrain_frame_psnr = calc_psnr(hr_frame, pretrain_frame)
                avg_sr_frame_psnr = calc_psnr(hr_frame, sr_frame)
                avg_bicubic_frame_ssim = ssim(hr_frame, bicubic_frame, multichannel=True)
                avg_pretrain_frame_ssim = ssim(hr_frame, pretrain_frame, multichannel=True)
                avg_sr_frame_ssim = ssim(hr_frame, sr_frame, multichannel=True)


                psnr_list = np.array([avg_sr_frame_psnr, avg_pretrain_frame_psnr, avg_bicubic_frame_psnr])
                max_idx = psnr_list.argmax()
                print(f'[pts: {pts}, metric: psnr] ' +
                      f'{"*" if max_idx == 0 else ""}online: {avg_sr_frame_psnr}, ' +
                      f'{"*" if max_idx == 1 else ""}pretrain: {avg_pretrain_frame_psnr}, ' +
                      f'{"*" if max_idx == 2 else ""}bicubic: {avg_bicubic_frame_psnr}')

                ssim_list = np.array([avg_sr_frame_ssim, avg_pretrain_frame_ssim, avg_bicubic_frame_ssim])
                max_idx = ssim_list.argmax()
                print(f'[pts: {pts}, metric: ssim] ' +
                      f'{"*" if max_idx == 0 else ""}online: {avg_sr_frame_ssim}, ' +
                      f'{"*" if max_idx == 1 else ""}pretrain: {avg_pretrain_frame_ssim}, ' +
                      f'{"*" if max_idx == 2 else ""}bicubic: {avg_bicubic_frame_ssim}')

                result = {
                    'pts': pts,
                    'psnr': {
                        'bicubic': avg_bicubic_frame_psnr,
                        'pretrain': avg_pretrain_frame_psnr,
                        'online': avg_sr_frame_psnr,
                    },
                    'ssim': {
                        'bicubic': avg_bicubic_frame_ssim,
                        'pretrain': avg_pretrain_frame_ssim,
                        'online': avg_sr_frame_ssim,
                    }
                }
                result_file.write(json.dumps(result) + ',\n')
                result_file.flush()

                pts += 1

            if pts>30:
                sleeptime = 50
            await asyncio.sleep(sleeptime)

    # video.release()

    loop = asyncio.get_event_loop()
    loop.create_task(show_frame())
    loop.run_forever()


def learn(state_queue: mp.Queue, patch_queue: mp.Queue,online_dataset):
    torch.cuda.set_device(1)
    # model = load_model()
    # model.cuda()

    # r_selector = RandomSelector()
    # for i in range(args.batch_size * 10):
    #     online_dataset.put(*r_selector.select_patch())

    r_selector = RandomSelector2()
    for lr,hr in r_selector.select_patches():
        online_dataset.put(lr, hr)
    #
    # for i in range(640//4):
    #     for lr,hr in r_selector.select_patch():
    #         online_dataset.put(lr,hr)

    # selector = Selector()
    selector = CandidatesSelector()
    # loader = DataLoader(dataset=online_dataset, num_workers=1, persistent_workers=True,
    #                     batch_size=args.batch_size, pin_memory=False, shuffle=True)
    cap = cv2.VideoCapture()
    cap.open(args.hr_video)

    async def update_dataset():
        print("update dataset!!")
        count = 0
        while True:
            reference_frame = []
            hr_s = []
            co = 4
            # print("before",patch_queue.qsize())
            if patch_queue.qsize() >= 2:
                while patch_queue.qsize() > 0 and co >0:
                    reference_frame.append(patch_queue.get())
                    count += 1
                    co -= 1
                # reference_frame.append(patch_queue.get())
                # print("after",patch_queue.qsize())
                # count += 1
                # ret, hr_frame = cap.read()
                # hr_frame = cv2.cvtColor(hr_frame, cv2.COLOR_BGR2RGB)
                # for _ in range(args.sample_interval - 1):
                #     _, _ = cap.read()
                # hr_s.append(hr_frame)
                print("count",count)
                patches = selector.select_patches(reference_frame, count, hr_s)
                # Image.fromarray(frame).save("selector_patches/frame.png")
                # Image.fromarray(hr_frame).save("selector_patches/hr_frame.png")

                for lr, hr in patches:
                    online_dataset.put(lr, hr)
                print("update once!....")
#--------------- train model
            # reference_frame = []
            # hr_s = []
            # # patches =[]
            # if not patch_queue.empty():
            #     # index = 3
            #     # while patch_queue.qsize() > 0 and index>0:
            #     reference_frame.append(patch_queue.get())
            #         # print(patch_queue.qsize())
            #     count += 1
            #         # index -= 1
            #     print("count",count)
            #     patches = selector.select_patches(reference_frame,count,hr_s)
            #     # Image.fromarray(frame).save("selector_patches/frame.png")
            #     # Image.fromarray(hr_frame).save("selector_patches/hr_frame.png")
            #
            #     for lr, hr in patches:
            #         online_dataset.put(lr, hr)
            #     print("update once!....")
            #     print(online_dataset.__len__())
#---------------------------
            # reference_frame = []
            # # print(patch_queue.qsize())
            # if patch_queue.qsize() >= 2:
            #     while patch_queue.qsize() > 0:
            #         reference_frame.append(patch_queue.get())
            #         count += 1
            #     print("count : ", count)
            #     patches = selector.select_patches(reference_frame,count)
            #     for lr, hr in patches:
            #         online_dataset.put(lr, hr)
            #     print("update once!....")
            #     print(online_dataset.__len__())
            await asyncio.sleep(0)

    async def train():
        print("training!!")
        count = 0
        scale = 2
        while True:
            loader = DataLoader(dataset=online_dataset, num_workers=4, persistent_workers=True,
                                batch_size=args.batch_size*2, pin_memory=False, shuffle=True)
            model = load_model()
            model.cuda()
            print(len(online_dataset))

            # if state_queue.qsize() > 0:
            #     buffer = state_queue.get()
            #     if buffer:
            #         model.load_state_dict(torch.load(buffer))
            start = time.time()
            for _ in range(13):
            # for _ in range(3):
                train_one_epoch(model, loader, scale)
                # print("Epoch :",_)
            end = time.time()
            print("train time .....",end-start)
            print("Train ", count)
            count += 1
            # send state dict
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            model.zero_grad()

            buffer.seek(0)
            state_queue.put(buffer)
            await asyncio.sleep(0)

    loop = asyncio.get_event_loop()
    loop.create_task(update_dataset())
    loop.create_task(train())
    loop.run_forever()


def main():
    frame_queue = mp.Queue()
    patch_queue = mp.Queue()
    state_queue = mp.Queue()

    online_dataset = OnlineDataset(max_size=300) #x4
    # online_dataset = OnlineDataset(max_size=600) #x2
    # online_dataset = OnlineDataset(max_size=200) #x3

    # online_dataset = OnlineDataset(max_size=1024) #1024

    train_process = mp.Process(target=learn, args=(state_queue, patch_queue,online_dataset))
    train_process.start()

    render_process = mp.Process(target=render, args=(frame_queue, state_queue))
    render_process.start()

    # train_process = mp.Process(target=learn, args=(state_queue, patch_queue))
    # train_process.start()

    asyncio.run(run(frame_queue, patch_queue))

def getpatch_hash_table():
    table = []
    patch_size = 64
    scale = 4
    candidates = list(Path(args.candidates).iterdir())
    for candidate in candidates:
        l = len(list(Path(candidate).iterdir()))
        for i in range(0, l,10):
            # for i in range(max(0, pts - self.candidate_pts_range), pts):
            filename = candidate / f'{str(i).zfill(4)}.png'
            # lr_filename = "candidates_lr/" + str(candidate)[11:] + f'/{str(i).zfill(4)}.png'
            # print(filename)
            hr = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)
            # # lr = cv2.cvtColor(cv2.imread(str(lr_filename)), cv2.COLOR_BGR2RGB)
            lr = cv2.resize(hr, (480, 270), interpolation=cv2.INTER_CUBIC)
            patch_hash_table = json.loads(
                open("patch_hash_darts/" + str(candidate)[11:] + f'/{str(i).zfill(4)}.txt').read())
            for item in range(100):
                xy = patch_hash_table[item]['xy']

                lr_i = lr[xy[0]: xy[0] + patch_size, xy[1]:xy[1] + patch_size]
                p = cv2.img_hash.pHash(lr_i)[0]
                # hr_i = hr[xy[0] * scale:xy[0] * scale + patch_size * scale,
                #        xy[1] * scale:xy[1] * scale + patch_size * scale]
                table.append((filename, xy, p))
    print("finish creating patch hash table........",len(table))
    return table

if __name__ == '__main__':
    main()
