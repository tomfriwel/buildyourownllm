# 导入随机数生成模块
import random

# 设置随机数种子，保证每次运行结果一致（可以去掉此行获得随机结果）
random.seed(42)

# 定义生成文本的起始词和最大生成长度
prompt = "春江"  # 起始词
max_new_token = 100  # 最大生成的词数

# 读取文本文件内容
with open('ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 构建词汇表
chars = sorted(list(set(text)))  # 获取文本中所有独特的字符，并排序
vocab_size = len(chars)  # 词汇表大小
char_to_index = { ch:i for i,ch in enumerate(chars) }  # 字符到索引的映射
index_to_char = { i:ch for i,ch in enumerate(chars) }  # 索引到字符的映射
encode = lambda s: [char_to_index[c] for c in s]  # 将字符串编码为索引列表，例如 '春江' -> [0, 1]
decode = lambda l: ''.join([index_to_char[i] for i in l])  # 将索引列表解码为字符串，例如 [0, 1] -> '春江'

# 初始化转移矩阵，用于记录每个字符后出现的字符的次数
#      a    b    c  ... (vocab_size)
#   a  0    1    0  ...
#   b  0    0    3  ...
#   c  4    0    0  ...
#  ... ... ... ... ... ... ...
# (vocab_size)
# vocab_size * vocab_size的二维数组，记录每个词的下一个词的出现次数
transition = [[0 for _ in range(vocab_size)] for _ in range(vocab_size)]
# 如果vocab_size = 3，transition：transition = [
#     [0, 0, 0],  # 第一行
#     [0, 0, 0],  # 第二行
#     [0, 0, 0]   # 第三行
# ]

# 统计转移矩阵中的字符出现次数
for i in range(len(text) - 1):
    current_token_id = encode(text[i])[0]  # 当前字符的索引
    next_token_id = encode(text[i + 1])[0]  # 下一个字符的索引
    transition[current_token_id][next_token_id] += 1  # 更新转移矩阵

# 根据起始词生成新文本
generated_token = encode(prompt)  # 将起始词编码为索引列表

for i in range(max_new_token - 1):
    current_token_id = generated_token[-1]  # 获取当前词的索引
    logits = transition[current_token_id]  # 获取当前词的转移概率
    total = max(sum(logits), 1)  # 确保总和不为零
    # 归一化前 logits = [0, 10, ...., 298,..., 88, ..., 13, 0]
    #         len(logits) = vocab_size, sum(logits) = 6664
    # 归一化后 logits = [0/6664, 10/6664, ...., 298/6664,..., 88/6664, ..., 13/6664, 0/6664]
    #                = [0, 0.0015, ..., 0.0447, ..., 0.0132, ..., 0.00195, 0]
    logits = [logit / total for logit in logits]  # 归一化转移概率
    # 计算概率，随机采样，得到下一个词
    next_token_id = random.choices(range(vocab_size), weights=logits, k=1)[0]  # 根据概率随机选择下一个词
    generated_token.append(next_token_id)  # 将选择的词添加到生成结果中
    break

# 输出生成的文本
print(decode(generated_token))  # 将索引列表解码为字符串并打印