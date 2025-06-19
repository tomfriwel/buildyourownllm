import torch
import torch.nn as nn

# 定义一个线性层，输入特征维度为3，输出特征维度为2
linear_layer = nn.Linear(in_features=3, out_features=10)

print("线性层的权重:", linear_layer.weight)

# 创建一个输入张量，形状为 (1, 3)，表示1个样本，3个特征
input_tensor = torch.tensor([[1.0, 2.0, 3.0]])

# 通过线性层进行前向传播
output_tensor = linear_layer(input_tensor)

print("输入张量:", input_tensor)
print("输出张量:", output_tensor)