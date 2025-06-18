import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(42) # 随机数种子，方便复现

# 判断环境中是否有GPU
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
print(f"Using {device} device")

# 1. 创建tensor演示
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([2.0, 4.0, 6.0])

# 2. 基本运算演示
print(x + y)                # 加法: tensor([3., 6., 9.])
print(x * y)                # 点乘: tensor([2., 8., 18.])
print(torch.matmul(x, y))   # 矩阵乘法: tensor(28.)
print(x @ y)                # 另一种矩阵乘写法: tensor(28.)
print(x.shape)              # tensor的形状: torch.Size([3])

# 3. 定义模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 输入维度=1，输出维度=1
    
    def forward(self, x):
        return self.linear(x)

# 4. 生成训练数据
# 真实关系: y = 2x + 1
x_train = torch.rand(100, 1) * 10  # 生成 0-10 之间的随机数
y_train = 2 * x_train + 1 + torch.randn(100, 1) * 0.1  # 真实函数：y = 2x + 1 加上一些噪声
# 将数据移动到指定设备
x_train = x_train.to(device)
y_train = y_train.to(device)

# 5. 创建模型和优化器
# SGD (随机梯度下降)
# 均方误差损失（Mean Squared Error, MSE）
model = SimpleNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 6. 训练循环
epochs = 5000

print("\n训练开始...")
for epoch in range(epochs):
    # 前向传播，预测结果
    y_pred = model(x_train)
    
    # 计算预测值和真实值之间的损失
    loss = criterion(y_pred, y_train)
    
    # 反向传播，修改模型参数
    optimizer.zero_grad() # 清除旧的梯度
    loss.backward() # 计算新的梯度 
    optimizer.step() # 更新参数：参数 -= 学习率 * 梯度
    
    # 每100个epoch打印一次损失和参数
    if (epoch + 1) % 100 == 0:
        w = model.linear.weight.item()
        b = model.linear.bias.item()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, w: {w:.2f}, b: {b:.2f}')

# 7. 打印结果
w = model.linear.weight.item()
b = model.linear.bias.item()
print(f'\n训练完成！')
print(f'学习到的函数: y = {w:.2f}x + {b:.2f}')
print(f'实际函数: y = 2.00x + 1.00')

# 8. 测试模型
test_x = torch.tensor([[0.0], [5.0], [10.0]]).to(device)
with torch.no_grad():
    test_y = model(test_x)
    print("\n预测结果：")
    for x, y in zip(test_x, test_y):
        print(f'x = {x.item():.1f}, y = {y.item():.2f}')