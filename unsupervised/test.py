import torch
import numpy as np
# from run import compute
import os
# from run import *
from tqdm import tqdm
from info_nce import InfoNCE

nce_loss = InfoNCE()
a = torch.tensor([0, 0, 0, 1, 1, 2, 2])
b = torch.rand(7, 10)
c = torch.rand(3, 10)
d = torch.rand(10, 10)
all_loss = 0
for i in range(a[-1] + 1):
    all_loss = 0
    for j in b[a == i]:
        LOSS = nce_loss(j.unsqueeze(0), c[0].unsqueeze(0), d)
        all_loss += LOSS
    print(all_loss/(b[a==i]).sum())
