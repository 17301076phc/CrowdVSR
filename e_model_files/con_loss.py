import torch


def loss_fun(SR,HR):
    criterion_L1 = torch.nn.L1Loss().cuda()
    ''' SR Loss '''
    loss_SR = criterion_L1(SR, HR)

    ''' Consistency Loss '''
    SR_left_res = torch.nn.functional.interpolate(torch.abs(HR - SR), scale_factor=1 / scale, mode='bicubic', align_corners=False)
    SR_left_resT = torch.bmm(M_right_to_left.detach().contiguous().view(b * h, w, w), SR_right_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                             ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
    loss_cons = criterion_L1(SR_left_res * V_left.repeat(1, 3, 1, 1), SR_left_resT * V_left.repeat(1, 3, 1, 1))
    ''' Total Loss '''
    loss = loss_SR + 0.1 * loss_cons
    return  loss

