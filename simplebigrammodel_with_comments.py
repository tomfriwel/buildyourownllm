# 导入随机模块和类型提示模块
import random
from typing import List

# 设置随机种子，确保每次运行结果一致
random.seed(42) # 去掉此行，获得随机结果

# 定义一些初始的文本提示
prompts = ["春江", "往事"]
# 定义生成的最大新token数量
max_new_token = 100
# 定义最大迭代次数
max_iters = 8000
# 定义每次训练的批次大小
batch_size = 32
# 定义每个序列的最大长度
block_size = 8

# 读取训练数据文件内容
with open('ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 定义一个分词器类，用于将文本转化为数字表示，或将数字表示转化为文本
# 作用：# 1. 获取文本中所有的唯一字符，并创建字符到索引的映射
# 2. 将字符串转化为索引列表
# 3. 将索引列表转化为字符串
class Tokenizer:
    def __init__(self, text: str):
        # 获取文本中所有的唯一字符，并排序
        self.chars = sorted(list(set(text)))
        # 计算词汇表大小
        self.vocab_size = len(self.chars)
        # 创建字符到索引的映射
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        # 创建索引到字符的映射
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, s: str) -> List[int]:
        # 将字符串转化为索引列表
        return [self.stoi[c] for c in s]
    
    def decode(self, l: List[int]) -> str:
        # 将索引列表转化为字符串
        return ''.join([self.itos[i] for i in l])

# 定义一个简单的双字模型类
# 作用：# 1. 初始化转移概率矩阵
# 2. 前向传播方法，计算每个token的下一个token的概率分布
# 3. 生成新token的方法，根据输入序列生成新的token
class BigramLanguageModel():
    def __init__(self, vocab_size: int):
        # 初始化词汇表大小
        self.vocab_size = vocab_size

        # 初始化转移概率矩阵，用于存储每个字符到下一个字符的概率
        # eg: [[0, 0, 0, ...],
        #      [0, 0, 0, ...],
        #      [0, 0, 0, ...],
        #      ...]
        # len(transition) = vocab_size
        # len(transition[0]) = vocab_size
        # transition[i][j]表示字符i到字符j的转移概率
        # 初始化为0，表示还没有训练
        self.transition = [[0 for _ in range(vocab_size)] 
                          for _ in range(vocab_size)]
        
    def __call__(self, x):
        # 方便直接调用model(x)，等价于调用forward方法
        return self.forward(x)
    
    # 作用：# 1. 输入idx，是一个二维数组，表示多个序列
    # 2. 输出是一个三维数组，每个序列的每个token的下一个token的概率分布
    # 3. 每个token的下一个token的概率分布是一个一维数组，长度为词汇表大小
    def forward(self, list_of_tokens: List[List[int]]) -> List[List[List[float]]]:
        '''
        输入list_of_tokens，是一个二维数组，如[[1, 2, 3],
                                  [4, 5, 6]]
        表示同时希望推理的多个序列

        输出是一个三维数组，如[[[0.1, 0.2, 0.3, .. (vocab_size)],
                                [0.4, 0.5, 0.6, .. (vocab_size)],
                                [0.7, 0.8, 0.9, .. (vocab_size)]],

                               [[0.2, 0.3, 0.4, .. (vocab_size)],
                                [0.5, 0.6, 0.7, .. (vocab_size)],
                                [0.8, 0.9, 1.0, .. (vocab_size)]]]
        
        '''
        B = len(list_of_tokens)  # 批次大小
        T = len(list_of_tokens[0])  # 每一批的序列长度
        
        # 初始化logits，用于存储概率分布
        # logits是一个三维数组，形状为(B, T, vocab_size)
        # 由内向外len=vocab_size；len=T；len=B
        logits = [
            [[0.0 for _ in range(self.vocab_size)] 
             for _ in range(T)]
            for _ in range(B)
        ]
        # ----original version:
        for b in range(B):
            for t in range(T):
                current_token = list_of_tokens[b][t]
                # 计算了每个批次中，每一个token的下一个token的概率
                # len(logits[b][t]) = len(self.transition[current_token]) = vocab_size
                logits[b][t] = self.transition[current_token]
        
        # ----optimized version:
        # 只计算最后一个token的下一个token的概率，因为后面只会用到最后一个token的概率
        # for b in range(B):
        #     current_token = list_of_tokens[b][-1]
        #     # 计算了每个批次中，每一个token的下一个token的概率
        #     # len(logits[b][-1]) = len(self.transition[current_token]) = vocab_size
        #     logits[b][-1] = self.transition[current_token]
        return logits

    # 作用：# 1. 根据输入序列生成新的token，直到达到最大数量
    # 2. 每次生成一个token，并将其添加到序列中
    def generate(self, list_of_tokens: List[List[int]], max_new_tokens: int) -> List[int]:
        # 根据输入序列生成新的token，直到达到最大数量
        for _ in range(max_new_tokens):
            # 前向传播，计算每个token的下一个token的概率分布, eg: [[0.1, 0.2, 0.3, .. (vocab_size)],
            logits_batch = self(list_of_tokens)
            # logits_batch = BatchSize x CurrentTokenLength x VocabSize

            # batch_idx的长度就是批次大小=len(list_of_tokens)，也就是提示词的批次数，在这里batch_idx从0到1，也就是2个批次
            for batch_idx, logits in enumerate(logits_batch):
                # logits = CurrentTokenLength x VocabSize
                print(f"Batch {batch_idx}: logits length = {len(logits)} logits[0] length = {len(logits[0])}")
                # 我们计算了每一个token的下一个token的概率
                # 但实际上我们只需要最后一个token的“下一个token的概率”（可以优化，只计算最后一个token的概率）
                logits = logits[-1]
                total = max(sum(logits),1)
                # 归一化概率分布
                logits = [logit / total for logit in logits]
                # 根据概率随机采样下一个token
                next_token = random.choices(
                    range(self.vocab_size),
                    weights=logits,
                    k=1
                )[0]
                # 将采样的token添加到序列中
                list_of_tokens[batch_idx].append(next_token)
        return list_of_tokens

# 作用：# 1. 随机获取一批数据x和y用于训练
# 2. x和y都是二维数组，可以用于并行训练
# 3. 其中y数组内的每一个值，都是x数组内对应位置的值的下一个值
def get_batch(tokens, batch_size, block_size):
    '''
    随机获取一批数据x和y用于训练
    x和y都是二维数组，可以用于并行训练
    其中y数组内的每一个值，都是x数组内对应位置的值的下一个值
    格式如下：
    x = [[1, 2, 3],
         [9, 10, 11]]
    y = [[2, 3, 4],
         [10, 11, 12]]
    '''

    # block_size = 8
    # batch_size = 32
    # 随机选择batch_size个起始位置
    # batch_start_idx里都是0到len(tokens) - block_size之间的随机整数
    batch_start_idx = random.choices(range(len(tokens) - block_size), k=batch_size)
    # len(batch_start_idx) = batch_size = 32
    input_tokens, target_tokens = [], []

    # batch_size times
    # input_tokens = [[], [], ...]  # 每个子列表长度为block_size
    # len(input_tokens) = batch_size = 32
    # len(input_tokens[0]) = block_size = 8
    for i in batch_start_idx:
        input_tokens.append(tokens[i:i+block_size])
        target_tokens.append(tokens[i+1:i+block_size+1])
    return input_tokens, target_tokens

# 初始化分词器
tokenizer = Tokenizer(text)
# 获取词汇表大小
vocab_size = tokenizer.vocab_size

# 将文本转化为token序列，所有的字符都转化为数字表示, eg: "春江" -> [0, 1]
tokens_of_text = tokenizer.encode(text)

# 初始化语言模型
model = BigramLanguageModel(vocab_size)

# 训练模型
# 作用：# 1. 随机获取一批数据x和y用于训练
# 2. 更新转移概率矩阵
for iteration in range(max_iters):
    # 获取训练数据批次
    input_batch, target_batch = get_batch(tokens_of_text, batch_size, block_size)

    # len(input_batch) = batch_size = 32
    for batch_index in range(len(input_batch)):
        # len(input_batch[batch_index]) = block_size = 8
        for token_index in range(len(input_batch[batch_index])):
            input_token = input_batch[batch_index][token_index]
            target_token = target_batch[batch_index][token_index]
            # 更新转移概率矩阵
            model.transition[input_token][target_token] += 1
    # 每1000次迭代打印一次训练进度
    if iteration % 1000 == 0:
        print(f"Iteration {iteration}/{max_iters} completed.")

# 将提示文本转化为token序列
# eg: ["春江", "往事", "案头"] -> [[0, 1], [2, 3], [4, 5]]
prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]

# 使用模型生成新文本
result = model.generate(prompt_tokens, max_new_token)

# 解码生成的token序列并打印结果
for tokens in result:
    print(tokenizer.decode(tokens))
    print('-'*10)

'''
大致流程：
- 训练模型：计算每个token的下一个token的概率分布: get_batch(from tokens_of_text), transition
- 生成新文本：根据概率分布随机采样下一个token: generate, forward(based on transition)
'''