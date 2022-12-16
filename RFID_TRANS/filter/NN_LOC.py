import torch
from math import sqrt
from torch import nn
from torch.nn import Linear, ReLU


class LOCATION_NN(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self):
        super(LOCATION_NN,self).__init__()
        self.time_step = 50
        self.input_dim = 1        
        self.FC = Linear(self.time_step * self.input_dim, 10)
        self.relu = ReLU()
        self.FC2 = Linear(10, 2)
    
    def forward(self,x):
        x = x.view(-1, self.time_step * self.input_dim)
        output = self.FC(x)
        output = self.relu(output)
        output = self.FC2(output)
        
        return output


net = LOCATION_NN()
input = torch.randn(8, 50, 1)
out = net(input)
print(out.size())
