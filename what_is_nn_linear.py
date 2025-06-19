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

'''
比喻
Embedding 层：像一本字典，你输入一个单词（索引），它直接返回对应的定义（向量）。
线性层：像一个调音器，你输入一个音符（向量），它根据内部的调音规则（权重和偏置）生成一个新的音符（向量）。
两者的核心区别在于：

Embedding 层的输出是固定的，训练过程中更新的是索引对应的向量值。
线性层的输出是动态计算的，训练过程中更新的是权重矩阵和偏置。



嵌入层的输入是索引，输出是固定维度的嵌入向量，输入和输出的维度通常不一样。
线性层的输入和输出都是向量或张量，维度由 in_features 和 out_features 决定，通常也不一样。
'''