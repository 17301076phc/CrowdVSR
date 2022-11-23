import numpy as np
import torch.autograd.profiler as profiler

def str_to_time(s):
    if s.endswith('ms'):
        return float(s[:-2])*1e-3
    if s.endswith('us'):
        return float(s[:-2])*1e-6
    return float(s[:-1])

def speed(mode, input):
    dt = np.inf
    for _ in range(10):
        with profiler.profile(
                record_shapes=True, use_cuda=True
        ) as prof:
            with profiler.record_function('model_inference'):
                output = model(input)
                dt1 = str_to_time(
                    prof.key_averages().table(
                        sort_by='cpu_time_total', row_limit=10).split('CPU time total: ')[1].split('\n')[0])
                dt2 = str_to_time(
                    prof.key_averages().table(sort_by='cpu_time_total', row_limit=10).
                        split('CUDA time total: ')[1][:-1])
                dt = min(dt1+dt2, dt)

    pix = np.asarray(output.shape).prod()
    return pix / (dt * 1920*1080)