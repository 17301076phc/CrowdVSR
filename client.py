import asyncio
import io
import json
import time
from pathlib import Path

import scipy
from skimage.transform import resize
from PIL import Image
import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from aiortc import RTCPeerConnection, RTCSessionDescription
# from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from dataset import OnlineDataset
from imresize import imresize
from option import args
from process import load_model, load_e_model,train_one_epoch, train_one_epoch_emodel,inference,calc_psnr
from selector import Selector, RandomSelector, CandidatesSelector, RandomSelector2

setscale=2
ise=True

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
                # print(frame.shape) #3
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
    # pretrain_model = load_model()
    # model = load_model()
    pretrain_model = load_e_model()
    model = load_e_model()
    model.cuda()
    model.eval()

    # result_file = Path(f'result_{int(time.time())}.txt').open('w', encoding='utf8')
    # result_file = Path(f'result_org_x4.txt').open('w', encoding='utf8')
    # result_file = Path(f'result_x2.txt').open('w', encoding='utf8')
    # result_file = Path(f'result_org_x2.txt').open('w', encoding='utf8')
    # result_file = Path(f'result_x4.txt').open('w', encoding='utf8')
    # result_file = Path(f'result_org_x3.txt').open('w', encoding='utf8')
    result_file = Path(f'e_model_result_x2.txt').open('w', encoding='utf8')



    cap = cv2.VideoCapture()
    cap.open(args.hr_video)
    # video = cv2.VideoWriter("resultvideo.avi", cv2.VideoWriter_fourcc(*"MJPG"), 1,
    #                         (1920,1080))
    async def show_frame():
        pts = 0
        scale = setscale
        sleeptime = 60
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
                # bicubic_frame = imresize(frame.astype('uint8'),scalar_scale=2) # 自行实现的resize
                # img = Image.fromarray(frame.astype('uint8'))
                # bicubic_frame = np.array(img.resize((1920, 1080),resample=Image.BICUBIC))
                # bicubic_frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                sr_tensor= F.interpolate(
                    torch.from_numpy(frame/255).float().permute(2, 0, 1).unsqueeze(0),  # (1,3,W,H)
                    scale_factor=scale,
                    mode='bicubic',
                    align_corners=True)*255

                sr_tensor = sr_tensor.squeeze().permute(1, 2, 0)  # (W,H,3)
                bicubic_frame = sr_tensor.numpy()

                pretrain_frame = inference(pretrain_model, frame, scale,is_emodel=ise)
                sr_frame = inference(model, frame, scale,is_emodel=ise)

                # make video
                # cv2.imwrite("ttmp.png", sr_frame)
                # sframe = cv2.imread("ttmp.png")
                sframe = sr_frame[:, :, ::-1]

                # sframe = cv2.cvtColor(sr_frame, cv2.COLOR_BGR2RGB)
                print(sframe.shape[1],sframe.shape[0])
                # video.write(sframe)

                avg_bicubic_frame_psnr = calc_psnr(hr_frame, bicubic_frame,scale)
                avg_pretrain_frame_psnr = calc_psnr(hr_frame, pretrain_frame,scale)
                avg_sr_frame_psnr = calc_psnr(hr_frame, sr_frame,scale)
                avg_bicubic_frame_ssim = ssim(hr_frame, bicubic_frame, multichannel=True)
                avg_pretrain_frame_ssim = ssim(hr_frame, pretrain_frame, multichannel=True)
                avg_sr_frame_ssim = ssim(hr_frame, sr_frame, multichannel=True)

                avg_bicubic_frame_psnr = round(avg_bicubic_frame_psnr,3)
                avg_pretrain_frame_psnr = round(avg_pretrain_frame_psnr,3)
                avg_sr_frame_psnr = round(avg_sr_frame_psnr,3)

                avg_bicubic_frame_ssim = round(avg_bicubic_frame_ssim,3)
                avg_pretrain_frame_ssim = round(avg_pretrain_frame_ssim,3)
                avg_sr_frame_ssim = round(avg_sr_frame_ssim,3)

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
                sleeptime = 70
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

    r_selector = RandomSelector2(scale=setscale)
    for lr,hr in r_selector.select_patches():
        online_dataset.put(lr, hr)

    # for i in range(640//4):
    #     for lr,hr in r_selector.select_patch():
    #         online_dataset.put(lr,hr)

    # selector = Selector()
    selector = CandidatesSelector(scale=setscale)
    # loader = DataLoader(dataset=online_dataset, num_workers=1, persistent_workers=True,
    #                     batch_size=args.batch_size, pin_memory=False, shuffle=True)
    # cap = cv2.VideoCapture()
    # cap.open(args.hr_video)

    async def update_dataset():
        print("update dataset!!")
        count = 0
        while True:
# ---------------------------------------ours------------------
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
                # patches = selector.select_patches(reference_frame, count, hr_s)
                patches = selector.online_data_select_patches(reference_frame, count, hr_s)
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
        scale = setscale
        while True:
            loader = DataLoader(dataset=online_dataset, num_workers=4, persistent_workers=True,
                                batch_size=args.batch_size*2, pin_memory=False, shuffle=True)
            # model = load_model()
            model = load_e_model()
            model.cuda()
            print(len(online_dataset))

            # if state_queue.qsize() > 0:
            #     buffer = state_queue.get()
            #     if buffer:
            #         model.load_state_dict(torch.load(buffer))
            start = time.time()
            for _ in range(20):
            # for _ in range(3):
            #     train_one_epoch(model, loader, scale)
                train_one_epoch_emodel(model, loader, scale)

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

    # online_dataset = OnlineDataset(max_size=1024) #org

    train_process = mp.Process(target=learn, args=(state_queue, patch_queue,online_dataset))
    train_process.start()

    render_process = mp.Process(target=render, args=(frame_queue, state_queue))
    render_process.start()

    # train_process = mp.Process(target=learn, args=(state_queue, patch_queue))
    # train_process.start()

    asyncio.run(run(frame_queue, patch_queue))

if __name__ == '__main__':
    main()
