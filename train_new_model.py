import json
import math
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import OnlineDataset
from new_model import OurNetwork

from process import load_model, train_one_epoch,quantize
from selector import RandomSelector
from option import args
import pretrain_dataloader as pre_dataloader


network_config = {4: {'block': 12, 'feature': 48},
                  3: {'block': 8, 'feature': 42},
                  2: {'block': 8, 'feature': 26},
                  1: {'block': 1, 'feature': 26}}

def psnr(sr, hr, scale=4, rgb_range=255, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range

    mse = diff.pow(2).mean()

    return -10 * math.log10(mse)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model = OurNetwork(network_config)
model.set_target_scale(4)
# device_ids = [0, 1, 2]
# model = torch.nn.DataParallel(model, device_ids=device_ids) # 指定要用到的设备

model = model.to(torch.device('cuda'))


pre_loader = pre_dataloader.Data()
# train_loader = pre_loader.loader_train
# test_loader = pre_loader.loader_test
div = pre_loader.div_loader

# online_dataset = OnlineDataset(max_size=64000)

# r_selector = RandomSelector()
print("preparing data....")
# print(r_selector.candidates)
# for i in range(50000):
#     online_dataset.put(*r_selector.select_patch())
# loader = DataLoader(dataset=online_dataset, num_workers=4,batch_size=64,  pin_memory=True, shuffle=True)

print("Start pretraining....")
print(model.networks)
result_file = Path(f'result_our_pretrain.txt').open('w', encoding='utf8')

for _ in range(100):
    # model.set_target_scale(4)

    model.train()
    loss_fn = nn.L1Loss()
    # loss_fn = nn.MSELoss()

    avg_loss = []
    optimizer = torch.optim.Adam(model.networks[4 - 1].parameters(), lr=1e-4, weight_decay=0)

    for data in div:
        input_tensor, target_tensor = data[0].cuda(), data[1].cuda()
        optimizer.zero_grad()
        output_tensor = model(input_tensor)
        loss = loss_fn(output_tensor, target_tensor)
        avg_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        # if iteration%20 == 0:
    print("Epoch ",_)
    print("Loss: ", sum(avg_loss)/len(avg_loss))

    # avg_psnr = []
    # model.eval()
    # for data in test_loader:
    #     input_tensor, target_tensor = data[0].cuda(), data[1].cuda()
    #
    #     with torch.no_grad():
    #         output_tensor = model(input_tensor)
    #     output_tensor = quantize(output_tensor)
    #     tmp = psnr(output_tensor, target_tensor)
    #     avg_psnr.append(float(tmp))
    #
    # print( 'Test data psnr ', sum(avg_psnr)/len(avg_psnr))

    result = {
        'Epoch': _,
        'loss': sum(avg_loss)/len(avg_loss)
    }
    result_file.write(json.dumps(result) + ',\n')
    result_file.flush()



print("Saving model......")
torch.save(model.state_dict(), "our_model/div2k_x4_our.pth")
print("Finished !....")


