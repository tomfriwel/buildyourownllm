import torch
import torch.nn as nn

# 定义一个嵌入层，颜色类别大小为5，嵌入维度为3（对应RGB值）
embedding = nn.Embedding(5, 3) # 5种颜色，每种颜色用3个值（R, G, B）表示

print(embedding)

# 输入是一个颜色类别的索引序列
input_indices = torch.tensor([0, 1, 2, 3, 4]) # 假设索引0-4分别表示红、绿、蓝、黄、紫

print("输入的颜色类别索引:", input_indices)

# 输出是对应的RGB嵌入向量
output = embedding(input_indices)
print("对应的RGB嵌入向量:", output)

# 嵌入层就像一个颜色查找表：
# 假设你有一个颜色类别的列表，每个颜色都有一个编号（索引），
# 而嵌入层的作用就是把这些编号映射到一个RGB值的向量。
# 这个向量可以看作是颜色的“位置”，表示它的RGB值。

# 类比：
# 如果把颜色类别比作颜色的名字，嵌入层就像是调色板。
# 每个颜色在调色板上都有一个具体的RGB值，
# 而这些RGB值是通过训练学习到的，能够反映颜色之间的关系。
# 比如，相近的颜色在RGB空间中会更接近，
# 同样，语义相近的颜色类别在嵌入空间中也会更接近。