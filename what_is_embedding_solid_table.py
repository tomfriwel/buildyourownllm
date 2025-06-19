import torch
import torch.nn as nn

# 定义一个固定的颜色RGB值映射表
color_to_rgb = {
    0: [255, 0, 0],   # 红色
    1: [0, 255, 0],   # 绿色
    2: [0, 0, 255],   # 蓝色
    3: [255, 255, 0], # 黄色
    4: [128, 0, 128]  # 紫色
}

# 输入是一个颜色类别的索引序列
# input_indices = torch.tensor([0, 1, 2, 3, 4]) # 假设索引0-4分别表示红、绿、蓝、黄、紫
input_indices = torch.tensor([0, 2]) # 假设索引0-4分别表示红、绿、蓝、黄、紫

print("输入的颜色类别索引:", input_indices)

# 输出是对应的RGB嵌入向量
output = torch.tensor([color_to_rgb[idx.item()] for idx in input_indices], dtype=torch.float32)
print("对应的RGB嵌入向量:", output)

# 嵌入层被替换为固定的颜色查找表：
# 假设你有一个颜色类别的列表，每个颜色都有一个编号（索引），
# 现在直接通过查找表获取对应的RGB值。