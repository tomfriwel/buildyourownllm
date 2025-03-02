import random
from typing import List

random.seed(42) # 去掉此行，获得随机结果

prompts = ["春江", "往事"]
max_new_token = 100
max_iters = 8000
batch_size = 32
block_size = 8

with open('ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

class Tokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]
    
    def decode(self, l: List[int]) -> str:
        return ''.join([self.itos[i] for i in l])

class BigramLanguageModel():
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

        self.transition = [[0 for _ in range(vocab_size)] 
                          for _ in range(vocab_size)]
        
    def __call__(self, x):
        # 方便直接调用model(x)
        return self.forward(x)
    
    def forward(self, idx: List[List[int]]) -> List[List[List[float]]]:
        '''
        输入idx，是一个二维数组，如[[1, 2, 3],
                                  [4, 5, 6]]
        表示同时希望推理的多个序列

        输出是一个三维数组，如[[[0.1, 0.2, 0.3, .. (vocab_size)],
                                [0.4, 0.5, 0.6, .. (vocab_size)],
                                [0.7, 0.8, 0.9, .. (vocab_size)]],

                               [[0.2, 0.3, 0.4, .. (vocab_size)],
                                [0.5, 0.6, 0.7, .. (vocab_size)],
                                [0.8, 0.9, 1.0, .. (vocab_size)]]]
        
        '''
        B = len(idx)  # 批次大小
        T = len(idx[0])  # 每一批的序列长度
        
        logits = [
            [[0.0 for _ in range(self.vocab_size)] 
             for _ in range(T)]
            for _ in range(B)
        ]
        
        for b in range(B):
            for t in range(T):
                current_token = idx[b][t]
                # 计算了每一个token的下一个token的概率
                logits[b][t] = self.transition[current_token]
                
        return logits

    def generate(self, idx: List[List[int]], max_new_tokens: int) -> List[int]:
        for _ in range(max_new_tokens):
            logits_batch = self(idx)
            for batch_idx, logits in enumerate(logits_batch):
                # 我们计算了每一个token的下一个token的概率
                # 但实际上我们只需要最后一个token的“下一个token的概率”
                logits = logits[-1]
                total = max(sum(logits),1)
                # 归一化
                logits = [logit / total for logit in logits]
                # 根据概率随机采样
                next_token = random.choices(
                    range(self.vocab_size),
                    weights=logits,
                    k=1
                )[0]
                idx[batch_idx].append(next_token)
        return idx
    
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
    ix = random.choices(range(len(tokens) - block_size), k=batch_size)
    x, y = [], []
    for i in ix:
        x.append(tokens[i:i+block_size])
        y.append(tokens[i+1:i+block_size+1])
    return x, y

tokenizer = Tokenizer(text)
vocab_size = tokenizer.vocab_size

tokens = tokenizer.encode(text)

model = BigramLanguageModel(vocab_size)

# 训练
for iter in range(max_iters):
    x_batch, y_batch = get_batch(tokens, batch_size, block_size)
    for i in range(len(x_batch)):
        for j in range(len(x_batch[i])):
            x = x_batch[i][j]
            y = y_batch[i][j]
            model.transition[x][y] += 1

prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]

# 推理
result = model.generate(prompt_tokens, max_new_token)

# decode
for tokens in result:
    print(tokenizer.decode(tokens))
    print('-'*10)