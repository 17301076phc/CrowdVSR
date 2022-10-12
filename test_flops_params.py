import torch
from thop import profile
from thop import clever_format
from model import MultiNetwork
network_config = {4: {'block': 8, 'feature': 48},
                  3: {'block': 8, 'feature': 42},
                  2: {'block': 8, 'feature': 26},
                  1: {'block': 1, 'feature': 26}}

net = MultiNetwork(network_config)
net.set_target_scale(4)

input = torch.randn(1,3,64,64)
flops, params = profile(net,(input,))

print("flops: ",flops, "params: ",params)
flops, params = clever_format([flops,params],"%.3f")

print("flops: ",flops, "params: ",params)