import torch
from torch import nn

head_size = 16 # 人为定义的注意力维度
x = torch.randn(1, 8, 16) # 单批次 (B=1), 8个token, 每个token 16维特征
B, T, C = x.shape # B=1, T=8, C=16

x = torch.randn(1, 8, 16)
print(x.shape)
print(x)
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)
q = query(x)
v = value(x)