import json
import math
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from e_model import edgeSR_TR,ENetwork
import dataloader as pre_dataloader


def psnr(sr, hr, scale=4, rgb_range=255, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    mse = diff.pow(2).mean()

    return -10 * math.log10(mse)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# s2 放大两倍 k7 卷积核 7 C7  通道数7
# model = edgeSR_TR("eSR-TR_s2_K7_C16")
# model = ENetwork("eSR-TR_s2_K3_C8")
# model = ENetwork("eSR-TR_s2_K3_C4")
# model = ENetwork("eSR-TR_s2_K3_C1")
# model = ENetwork("eSR-TR_s2_K3_C3")
model = ENetwork("eSR-TR_s2_K3_C2")


# device_ids = [0, 1, 2]
# model = torch.nn.DataParallel(model, device_ids=device_ids) # 指定要用到的设备

model = model.to(torch.device('cuda'))

pre_loader = pre_dataloader.Data()
# train_loader = pre_loader.loader_train
# test_loader = pre_loader.loader_test
div = pre_loader.div_loader

print("preparing data....")
result_file = Path(f'e_model_files/e_model/model.txt').open('w', encoding='utf8')
print(model)
result_file.write(str(model))
result_file.flush()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5)
print("Start pretraining....")
print(pre_loader.div_loader)
# result_file = Path(f'result_pretrain.txt').open('w', encoding='utf8')
loss_file = Path(f'e_model_loss.txt').open('w', encoding='utf8')
for _ in range(200):

    model.train()
    loss_fn = nn.L1Loss()
    # loss_fn = nn.SmoothL1Loss()
    avg_loss = []
    for data in div:
    # for data in sp:
        input_tensor, target_tensor = data[0].cuda(), data[1].cuda()
        optimizer.zero_grad()
        output_tensor = model(input_tensor)
        loss = loss_fn(output_tensor, target_tensor)
        avg_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    scheduler.step()
        # if iteration%20 == 0:
    print("Epoch ",_)
    print("Loss: ", sum(avg_loss)/len(avg_loss))
    print("lr rate:",optimizer.state_dict()['param_groups'][0]['lr'])
    torch.save(model.state_dict(), 'e_model_files/e_model/eSR-TR_s2_K3_C2.pth')
    result = {
        'Loss': sum(avg_loss)/len(avg_loss)
    }
    result_file.write(json.dumps(result) + ',\n')
    result_file.flush()


print("Saving model......")
torch.save(model.state_dict(), 'e_model_files/e_model/eSR-TR_s2_K3_C2.pth')
print("Finished !....")


