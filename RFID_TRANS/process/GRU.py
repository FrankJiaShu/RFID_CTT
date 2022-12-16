import torch
from torch import nn
from torch.nn import GRU, Linear
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义一个GRU模型


class KD_GRU(nn.Module):
    def __init__(self):
        super(KD_GRU, self).__init__()
        self.time_step = 50
        self.x_dim = 1
        self.h_dim = 60
        self.gru_layers = 1

        self.GRU_layer = GRU(self.x_dim, self.h_dim, self.gru_layers, batch_first=True)
        self.FC = Linear(self.h_dim, 2)

    def forward(self, x):
        batch, time_step, x_dim = x.size()
        h0 = torch.zeros((self.gru_layers, batch, self.h_dim)).to(device)
        output, hn = self.GRU_layer(x, h0)
        res = hn.squeeze(0)
        res = self.FC(res)
        return res


kd_gru = KD_GRU()
input = torch.randn(8, 50, 1)
output = kd_gru(input)
print(output.size())
target = np.random.rand(8, 2)
target = torch.from_numpy(target)
loss_fn = torch.nn.MSELoss(reduce=True, reduction='mean')
l = loss_fn(output, target)
print(l)