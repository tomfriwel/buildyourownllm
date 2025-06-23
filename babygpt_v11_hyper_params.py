import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
import time

torch.manual_seed(42)

prompts = ["春江", "往事"] # 推理的输入prompts，比如给AI一个开头，它会像接龙一样继续生成内容
max_new_token = 100 # 推理生成的最大tokens数量，就像规定接龙的字数上限
max_iters = 5000 # 训练的最大迭代次数，类似于学习的总课时数
eval_iters = 100 # 评估的迭代次数
eval_interval = 200 # 评估的间隔
batch_size = 64 # 每个批次的大小，像课堂上每次学习的学生人数
block_size = 256 # 每个序列的最大长度，类似于每次学习的内容长度
learning_rate = 3e-4 # 学习率，像学习的速度，太快可能记不住，太慢效率低
n_embed = 384 # 嵌入层的维度，类似于知识点的复杂程度
n_head = 6 # 多头注意力的头数，像一个团队中不同的专家同时关注不同的方面
n_layer = 6 # block的数量，类似于学习的章节数
tain_data_ratio = 0.9 # 训练数据占数据集的比例，像复习时用90%的时间学习，10%的时间测试
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

with open('ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

class Tokenizer:
    def __init__(self, text: str):
        # Tokenizer类用于将文本转化为数字和从数字还原文本，就像一本字典，帮助AI理解和生成内容
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> List[int]:
        # encode方法将字符串转化为数字序列，就像把文字翻译成密码
        return [self.stoi[c] for c in s]

    def decode(self, l: List[int]) -> str:
        # decode方法将数字序列还原为字符串，就像解码密码还原文字
        return ''.join([self.itos[i] for i in l])

class Block(nn.Module):
    def __init__(self, n_embed, n_head, dropout):
        # Block类是模型的基本单元，包含注意力机制和前馈网络，就像一个学习模块
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, dropout)
        self.ffwd = FeedFoward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed) # layer norm layer，像是对学习内容进行整理
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # forward方法定义了数据如何通过Block处理，就像学生如何学习知识
        x = x + self.sa(self.ln1(x)) # 使用了残差连接，保留原来的x信息，避免梯度消失，就像在学习新知识时保留旧知识的记忆
        x = x + self.ffwd(self.ln2(x))
        return x

class FeedFoward(nn.Module):
    def __init__(self, n_embed, dropout):
        # FeedFoward类是前馈网络，用于进一步处理数据，就像深入学习某个知识点
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.ReLU(), # 把负值变为0，正直不变的激活函数，就像过滤掉无用信息
            nn.Linear(n_embed * 4, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # forward方法定义了数据如何通过前馈网络处理
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, dropout):
        # MultiHeadAttention类实现了多头注意力机制，就像一个团队中不同的专家同时关注不同的方面
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed) # 投影层，把多头注意力的输出映射回n_embed维度
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # forward方法定义了数据如何通过多头注意力处理
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class Head(nn.Module):
    def __init__(self, head_size, dropout):
        # Head类是注意力机制的一个头，负责计算key、query和value
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # __init__里的module都会被pytorch自动当作layer来处理，用register_buffer后，这里就是一个普通的变量
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # forward方法定义了数据如何通过一个注意力头处理
        B, T, C = x.shape # (batch_size, block_size, n_embed)
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5) # (B, T, T)，最后缩放避免softmax过于稀疏
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # 上三角都是-inf，下三角是q和k的点积
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out
    
class BabyGPT(nn.Module):

    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        # BabyGPT类是一个简单的语言模型，就像一个会写诗的小学生
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # 嵌入层，把token映射到n_embd维空间，就像给每个字分配一个座位
        self.postion_embedding_table = nn.Embedding(block_size, n_embed) # 建设一个“位置”映射关系，就像给每个座位编号
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head, dropout=0.0) for _ in range(n_layer)]) # 多个Block组成的序列，就像多个学习模块
        self.ln_final = nn.LayerNorm(n_embed) # 最后的layer norm层，整理学习内容
        self.lm_head = nn.Linear(n_embd, vocab_size) # 线性层，把n_embd维空间映射到vocab_size维空间，就像把知识转化为答案

    def forward(self, idx, targets=None):
        # forward方法定义了数据如何通过BabyGPT处理，就像学生如何学习和回答问题
        B, T = idx.shape # B是batch size，T是block size
        T = min(T, self.block_size)
        idx = idx[:, -T:] # 不管输入的序列有多长，我们只取最后的block_size个token，就像只关注最近的内容
        tok_emb = self.token_embedding_table(idx) # 获得token的嵌入表示 (B,T,n_embd)
        pos_emb = self.postion_embedding_table(torch.arange(T, device=idx.device)) # 获得位置的嵌入表示 (T,n_embd)
        x = tok_emb + pos_emb # 给token的嵌入表示加上位置的嵌入表示，x有了“位置”信息！
        x = self.blocks(x)
        x = self.ln_final(x)
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
        # generate方法定义了如何生成新的token序列，就像学生根据已有内容继续写诗
        for _ in range(max_new_tokens):
            logits, _ = self(idx) # logits的形状是(B,T,vocab_size)，每一个token都计算了下一个token的概率
            logits = logits[:, -1, :] # 实际上我们只需要最后一个token算出来的值
            probs = F.softmax(logits, dim=-1) # 使用softmax函数算概率分布，这里dim=-1表示对最后一个维度进行softmax
            idx_next = torch.multinomial(probs, num_samples=1) # 根据概率分布随机采样，这里num_samples=1表示采样一个token
            idx = torch.cat((idx, idx_next), dim=1) # 把采样的token拼接到序列后面
        return idx

tokenizer = Tokenizer(text) # 初始化Tokenizer，就像准备一本字典
vocab_size = tokenizer.vocab_size # 计算词汇表大小

raw_data = torch.tensor(tokenizer.encode(text), dtype=torch.long).to(device) # 把文本转化为数字序列
n = int(tain_data_ratio*len(raw_data)) # 根据训练数据比例划分数据集

data = {'train': raw_data[:n], 'val': raw_data[n:]} # 定义训练集和验证集的数据，就像划分学习和测试的内容

def get_batch(data, batch_size, block_size):
    # get_batch函数用于生成训练数据的批次，就像从书中抽取一段内容进行学习
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, eval_iters):
    # estimate_loss函数用于评估模型的损失，就像测试学生的学习效果
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

model = BabyGPT(vocab_size, block_size, n_embed).to(device) # 初始化模型，就像准备一个学生开始学习

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # AdamW优化器，就像给学生制定学习计划

start_time = time.time() # 记录开始时间

for iter in range(max_iters):
    # 训练循环，就像学生每天学习一段时间
    x, y = get_batch(data['train'], batch_size, block_size) # 获取训练数据
    logits, loss = model(x, y) # 模型学习并计算损失
    optimizer.zero_grad(set_to_none=True) # 清空梯度，就像每天学习前清空桌面
    loss.backward() # 计算梯度，就像学生复习错误的地方
    optimizer.step() # 更新模型参数，就像学生改正错误

    tokens_processed += batch_size * block_size # 记录处理的token数量

    if iter % eval_interval == 0:
        elapsed = time.time() - start_time
        tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
        losses = estimate_loss(model, data, batch_size, block_size, eval_iters)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, speed: {tokens_per_sec:.2f} tokens/sec")

prompt_tokens = torch.stack([torch.tensor(tokenizer.encode(p)).to(device) for p in prompts])

result = model.generate(prompt_tokens, max_new_token)

for tokens in result:
    print(tokenizer.decode(tokens.tolist()))
    print('-'*10)