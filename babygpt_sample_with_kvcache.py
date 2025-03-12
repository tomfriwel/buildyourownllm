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

    def forward(self, x, kv_cache=None):
        # 使用了残差连接，保留原来的x信息，避免梯度消失
        sa_out, new_kv_cache = self.sa(self.ln1(x), kv_cache)
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, new_kv_cache

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

    def forward(self, x, kv_cache=None):
        outputs = []
        new_kv_caches = []
        
        # 处理每个注意力头
        for i, head in enumerate(self.heads):
            # 获取当前头的kv缓存
            head_kv_cache = None if kv_cache is None else kv_cache[i]
            # 计算当前头的输出和新的kv缓存
            out, new_head_kv_cache = head(x, head_kv_cache)
            outputs.append(out)
            new_kv_caches.append(new_head_kv_cache)
        
        # 连接所有头的输出
        out = torch.cat(outputs, dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out, new_kv_caches

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # __init__里的module都会被pytorch自动当作layer来处理，用register_buffer后，这里就是一个普通的变量
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape # (batch_size, block_size, n_embed)
        
        # 计算当前输入的k和v
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        
        # 如果有kv缓存，将当前的k和v与缓存连接起来
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)  # 连接时间维度
            v = torch.cat([v_cache, v], dim=1)  # 连接时间维度
        
        # 存储当前的k和v作为新的缓存
        new_kv_cache = (k, v)
        
        # 计算attention
        # q: (B, T, head_size), k: (B, T', head_size) -> (B, T, T')
        wei = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)
        
        # 获取注意力掩码的尺寸
        T_total = k.size(1)  # 总的序列长度(包括缓存)
        T_current = q.size(1) # 当前输入的序列长度
        
        # 创建掩码，确保当前token只关注过去的token
        # 注意掩码大小需要匹配 wei 的最后两个维度 (T_current, T_total)
        mask = torch.tril(torch.ones(T_total, T_total, device=x.device))
        mask = mask[-T_current:, :]  # 只取最后T_current行
        
        # 应用掩码
        wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T_current, T_total)
        wei = self.dropout(wei)
        
        # 计算输出
        out = wei @ v # (B, T_current, T_total) @ (B, T_total, head_size) = (B, T_current, head_size)
        
        return out, new_kv_cache
    
class BabyGPT(nn.Module):

    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # 嵌入层，把token映射到n_embd维空间
        self.postion_embedding_table = nn.Embedding(block_size, n_embed) # 建设一个"位置"映射关系
        # 改用ModuleList以便单独处理每个block的kv_cache
        self.blocks = nn.ModuleList([Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embd, vocab_size) # 线性层，把n_embd维空间映射到vocab_size维空间，

    def forward(self, idx, targets=None, kv_cache=None):
        B, T = idx.shape # B是batch size，T是block size
        
        # 处理kv_cache，确定位置偏移量
        if kv_cache is not None:
            # 如果有缓存，我们只需要处理最新的token
            pos_offset = kv_cache[0][0][0][0].size(1) if kv_cache else 0
            idx = idx[:, -1:]  # 只处理最后一个token
            T = 1
        else:
            pos_offset = 0
            T = min(T, self.block_size)
            idx = idx[:, -T:] # 不管输入的序列有多长，我们只取最后的block_size个token
            
        tok_emb = self.token_embedding_table(idx) # 获得token的嵌入表示 (B,T,n_embd)
        pos_emb = self.postion_embedding_table(torch.arange(pos_offset, pos_offset + T, device=idx.device)) # 获得位置的嵌入表示 (T,n_embd)
        x = tok_emb + pos_emb # 给token的嵌入表示加上位置的嵌入表示，x有了"位置"信息！
        
        # 处理每个block，并且维护kv_cache
        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            block_kv_cache = None if kv_cache is None else kv_cache[i]
            x, block_new_kv_cache = block(x, block_kv_cache)
            new_kv_cache.append(block_new_kv_cache)
            
        x = self.ln_final(x)
        logits = self.lm_head(x) # 通过线性层，把embedding结果重新映射回vocab_size维空间 (B,T,vocab_size)

        if targets is None: # 推理场景，不需要计算损失值
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # 把(B,T,C)的形状转换为(B*T,C)
            targets = targets.view(B*T) # 把(B,T)的形状转换为(B*T)
            loss = F.cross_entropy(logits, targets) # 计算交叉熵损失
        return logits, loss, new_kv_cache

    def generate(self, idx, max_new_tokens):
        # 初始化kv_cache为None
        kv_cache = None
        
        for _ in range(max_new_tokens):
            # 将kv_cache传入forward
            logits, _, kv_cache = self(idx, kv_cache=kv_cache)
            
            # 只需要关注最后一个token的预测
            logits = logits[:, -1, :] # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        
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