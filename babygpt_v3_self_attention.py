import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
import time

torch.manual_seed(42)

# 比喻：就像我们在写诗时，先有一些关键词（春江、往事），然后根据这些关键词生成一整首诗。
prompts = ["春江", "往事"] # 推理的输入prompts

# 比喻：就像我们在学习时，设定一个目标，比如最多学习100个新单词。
max_new_token = 100 # 推理生成的最大tokens数量

# 比喻：就像我们在健身时，设定一个训练计划，比如最多训练5000次。
max_iters = 5000 # 训练的最大迭代次数

# 比喻：就像我们在考试时，设定一个评估标准，比如每隔200分钟评估一次。
eval_interval = 200 # 评估的间隔

# 比喻：就像我们在做饭时，设定每次做饭的份量，比如每次做32份。
batch_size = 32 # 每个批次的大小

# 比喻：就像我们在写文章时，设定每段的最大长度，比如每段最多8句话。
block_size = 8 # 每个序列的最大长度

# 比喻：就像我们在学习时，设定学习的深度，比如学习的知识点维度是32。
n_embed = 32 # 嵌入层的维度

# 比喻：就像我们在分配时间时，设定训练和验证的比例，比如90%的时间用于训练，10%的时间用于验证。
tain_data_ratio = 0.9 # 训练数据占数据集的比例，剩下的是验证数据

# 比喻：就像我们选择工具时，优先选择最快的工具，比如选择GPU或MPS。
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

# 比喻：就像我们在阅读一本书时，先把书的内容全部读入内存。
with open('ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 比喻：就像我们在语言学习时，先建立一个词汇表，把每个单词和它的编号对应起来。
class Tokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    # 比喻：就像我们把一句话翻译成编号序列，方便计算机处理。
    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    # 比喻：就像我们把编号序列翻译回原来的文字，方便人类阅读。
    def decode(self, l: List[int]) -> str:
        return ''.join([self.itos[i] for i in l])

# 比喻：就像我们在团队中分工，每个人负责不同的任务，比如有的人负责找关键词，有的人负责提问，有的人负责回答。
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # __init__里的module都会被pytorch自动当作layer来处理，用register_buffer后，这里就是一个普通的变量
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    # 比喻：就像我们在开会时，每个人都带着自己的任务（关键词、问题、答案）进入会议室。
    def forward(self, x):
        B, T, C = x.shape # (batch_size, block_size, n_embed)
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5) # (B, T, head_size) @ (B, head_size, T) = (B, T, T)，最后缩放避免softmax过于稀疏
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # 上三角都是-inf，下三角是q和k的点积
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        out = wei @ v # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out
    
class BabyGPT(nn.Module):

    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # 嵌入层，把token映射到n_embd维空间
        self.postion_embedding_table = nn.Embedding(block_size, n_embed) # 建设一个“位置”映射关系
        self.sa_head = Head(n_embed) # self-attention 头
        self.lm_head = nn.Linear(n_embd, vocab_size) # 线性层，把n_embd维空间映射到vocab_size维空间，

    def forward(self, idx, targets=None):
        B, T = idx.shape # B是batch size，T是block size
        T = min(T, self.block_size)
        idx = idx[:, -T:] # 不管输入的序列有多长，我们只取最后的block_size个token
        tok_emb = self.token_embedding_table(idx) # 获得token的嵌入表示 (B,T,n_embd)
        pos_emb = self.postion_embedding_table(torch.arange(T, device=idx.device)) # 获得位置的嵌入表示 (T,n_embd)
        x = tok_emb + pos_emb # 给token的嵌入表示加上位置的嵌入表示，x有了“位置”信息！
        x = self.sa_head(x) # self-attention
        logits = self.lm_head(x) # 通过线性层，把embedding结果重新映射回vocab_size维空间 (B,T,vocab_size)

        if targets is None: # 推理场景，不需要计算损失值
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # 把(B,T,C)的形状转换为(B*T,C)，因为交叉熵损失函数第一个参数只接受二维输入。这个操作并没有丢失信息
            targets = targets.view(B*T) # 把(B,T)的形状转换为(B*T)，因为交叉熵损失函数第二个参数只接受一维输入。这个操作并没有丢失信息
            loss = F.cross_entropy(logits, targets) # 计算交叉熵损失
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # 比喻：就像我们根据关键词不断续写诗句，直到达到预定的字数。
        for _ in range(max_new_tokens):
            logits, _ = self(idx) # logits的形状是(B,T,vocab_size)，每一个token都计算了下一个token的概率
            logits = logits[:, -1, :] # 实际上我们只需要最后一个token算出来的值
            probs = F.softmax(logits, dim=-1) # 使用softmax函数算概率分布，这里dim=-1表示对最后一个维度进行softmax
            idx_next = torch.multinomial(probs, num_samples=1) # 根据概率分布随机采样，这里num_samples=1表示采样一个token
            idx = torch.cat((idx, idx_next), dim=1) # 把采样的token拼接到序列后面
        return idx

tokenizer = Tokenizer(text)
vocab_size = tokenizer.vocab_size

raw_data = torch.tensor(tokenizer.encode(text), dtype=torch.long).to(device)
n = int(tain_data_ratio*len(raw_data))
data = {'train': raw_data[:n], 'val': raw_data[n:]}

def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, eval_iters):
    '''
    计算模型在训练集和验证集上的损失
    '''
    out = {}
    model.eval() # 切换到评估模式
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data[split], batch_size, block_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # 切换回训练模式
    return out

model = BabyGPT(vocab_size, block_size, n_embed).to(device)

# 定义学习率和评估迭代次数
learning_rate = 1e-2 # 学习率
eval_iters = 100 # 评估的迭代次数

# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start_time = time.time()
tokens_processed = 0

for iter in range(max_iters):
    x, y = get_batch(data['train'], batch_size, block_size)
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    tokens_processed += batch_size * block_size

    if iter % eval_interval == 0:
        elapsed = time.time() - start_time
        tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
        losses = estimate_loss(model, data, batch_size, block_size, eval_iters)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, speed: {tokens_per_sec:.2f} tokens/sec")

# 推理
prompt_tokens = torch.stack([torch.tensor(tokenizer.encode(p)).to(device) for p in prompts])

# 生成
result = model.generate(prompt_tokens, max_new_token)

# 解码并打印结果
for tokens in result:
    print(tokenizer.decode(tokens.tolist()))
    print('-'*10)