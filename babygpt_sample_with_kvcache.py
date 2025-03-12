import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
import time

torch.manual_seed(42)

max_new_token = 100 # 推理生成的最大tokens数量
max_iters = 5000 # 训练的最大迭代次数
eval_iters = 100 # 评估的迭代次数
eval_interval = 50 # 评估的间隔
batch_size = 64 # 每个批次的大小
block_size = 256 # 每个序列的最大长度
learning_rate = 3e-4 # 学习率
n_embed = 384 # 嵌入层的维度
n_head = 6 # 多头注意力的头数
n_layer = 6 # block的数量
dropout = 0.2 # dropout的比例
tain_data_ratio = 0.9 # 训练数据占数据集的比例，剩下的是验证数据
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

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

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed) # layer norm layer
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # 使用了残差连接，保留原来的x信息，避免梯度消失
        x = x + self.ffwd(self.ln2(x))
        return x

class FeedFoward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.ReLU(), # 把负值变为0，正直不变的激活函数
            nn.Linear(n_embed * 4, n_embed),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed) # 投影层，把多头注意力的输出映射回n_embed维度
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # __init__里的module都会被pytorch自动当作layer来处理，用register_buffer后，这里就是一个普通的变量
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # (batch_size, block_size, n_embed)
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5) # (B, T, head_size) @ (B, head_size, T) = (B, T, T)，最后缩放避免softmax过于稀疏
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # 上三角都是-inf，下三角是q和k的点积
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out
    
class BabyGPT(nn.Module):

    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # 嵌入层，把token映射到n_embd维空间
        self.postion_embedding_table = nn.Embedding(block_size, n_embed) # 建设一个“位置”映射关系
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embd, vocab_size) # 线性层，把n_embd维空间映射到vocab_size维空间，

    def forward(self, idx, targets=None):
        B, T = idx.shape # B是batch size，T是block size
        T = min(T, self.block_size)
        idx = idx[:, -T:] # 不管输入的序列有多长，我们只取最后的block_size个token
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
        for _ in range(max_new_tokens):
            logits, _ = self(idx) # logits的形状是(B,T,vocab_size)，每一个token都计算了下一个token的概率
            logits = logits[:, -1, :] # 实际上我们只需要最后一个token算出来的值
            probs = F.softmax(logits, dim=-1) # 使用softmax函数算概率分布，这里dim=-1表示对最后一个维度进行softmax
            idx_next = torch.multinomial(probs, num_samples=1) # 根据概率分布随机采样，这里num_samples=1表示采样一个token
            idx = torch.cat((idx, idx_next), dim=1) # 把采样的token拼接到序列后面
        return idx
    
tokenizer = Tokenizer(text)
vocab_size = tokenizer.vocab_size

model = BabyGPT(vocab_size, block_size, n_embed).to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

while True:
    prompt = input("请输入文字: ")
    prompt_tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    start_time = time.time()
    result = model.generate(prompt_tokens, max_new_token)
    end_time = time.time()

    elapsed_time = end_time - start_time
    tokens_per_second = max_new_token / elapsed_time

    print(tokenizer.decode(result[0].tolist()))
    print(f"> 生成速度: {tokens_per_second:.2f} tokens/s")
    print('-'*10)