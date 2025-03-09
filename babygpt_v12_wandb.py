import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
import time
import wandb

torch.manual_seed(42)

prompts = ["春江", "往事"] # 推理的输入prompts
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

wandb.init(
    project="babygpt",
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "block_size": block_size,
        "n_embed": n_embed,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
    }
)

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
        elapsed_mins = elapsed // 60
        elapsed_secs = elapsed % 60
        wandb.log({
            "train_loss": losses['train'],
            "val_loss": losses['val'],
            "tokens_per_sec": tokens_per_sec,
            "iteration": iter
        })
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, speed: {tokens_per_sec:.2f} tokens/sec, time: {int(elapsed_mins)}m {elapsed_secs:.1f}s")

prompt_tokens = torch.stack([torch.tensor(tokenizer.encode(p)).to(device) for p in prompts])

result = model.generate(prompt_tokens, max_new_token)

for tokens in result:
    print(tokenizer.decode(tokens.tolist()))
    print('-'*10)

save_path = 'model.pth'
torch.save(model.state_dict(), save_path)
print(f"模型已保存到{save_path}")
'''
simpx@ThePC:~/buildyourownllm$ python babygpt_v12_wandb.py
wandb: Using wandb-core as the SDK backend. Please refer to https://wandb.me/wandb-core for more information.
wandb: Currently logged in as: simpxx (simpxx-zhejiang-university). Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /home/simpx/buildyourownllm/wandb/run-20250309_235239-ysgr3tei
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run absurd-frog-1
wandb: ⭐️ View project at https://wandb.ai/simpxx-zhejiang-university/babygpt
wandb: 🚀 View run at https://wandb.ai/simpxx-zhejiang-university/babygpt/runs/ysgr3tei
step 0: train loss 8.0529, val loss 8.0512, speed: 55304.00 tokens/sec, time: 0m 0.3s
step 50: train loss 5.9337, val loss 6.0072, speed: 102707.49 tokens/sec, time: 0m 8.1s
step 100: train loss 5.7529, val loss 5.8782, speed: 104971.14 tokens/sec, time: 0m 15.8s
step 150: train loss 5.4843, val loss 5.6578, speed: 105831.56 tokens/sec, time: 0m 23.4s
step 200: train loss 5.2826, val loss 5.4927, speed: 106169.27 tokens/sec, time: 0m 31.0s
step 250: train loss 5.1371, val loss 5.3766, speed: 105984.49 tokens/sec, time: 0m 38.8s
step 300: train loss 5.0116, val loss 5.2703, speed: 105957.57 tokens/sec, time: 0m 46.5s
step 350: train loss 4.9237, val loss 5.1528, speed: 106056.41 tokens/sec, time: 0m 54.2s
step 400: train loss 4.8080, val loss 5.0865, speed: 105914.43 tokens/sec, time: 1m 2.0s
step 450: train loss 4.7279, val loss 4.9910, speed: 105835.10 tokens/sec, time: 1m 9.8s
step 500: train loss 4.6646, val loss 4.9363, speed: 105850.83 tokens/sec, time: 1m 17.5s
step 550: train loss 4.6004, val loss 4.8573, speed: 105825.23 tokens/sec, time: 1m 25.3s
step 600: train loss 4.5383, val loss 4.8317, speed: 105717.62 tokens/sec, time: 1m 33.1s
step 650: train loss 4.4883, val loss 4.7752, speed: 105713.79 tokens/sec, time: 1m 40.9s
step 700: train loss 4.4415, val loss 4.7334, speed: 105657.30 tokens/sec, time: 1m 48.7s
step 750: train loss 4.4077, val loss 4.7024, speed: 105533.37 tokens/sec, time: 1m 56.6s
step 800: train loss 4.3546, val loss 4.6546, speed: 105446.23 tokens/sec, time: 2m 4.5s
step 850: train loss 4.3154, val loss 4.6399, speed: 105418.73 tokens/sec, time: 2m 12.3s
step 900: train loss 4.2720, val loss 4.5936, speed: 105405.89 tokens/sec, time: 2m 20.0s
step 950: train loss 4.2308, val loss 4.5587, speed: 105314.04 tokens/sec, time: 2m 27.9s
step 1000: train loss 4.1714, val loss 4.5141, speed: 105286.65 tokens/sec, time: 2m 35.8s
step 1050: train loss 4.1327, val loss 4.4774, speed: 105286.05 tokens/sec, time: 2m 43.6s
step 1100: train loss 4.1021, val loss 4.4610, speed: 105221.62 tokens/sec, time: 2m 51.4s
step 1150: train loss 4.0632, val loss 4.4143, speed: 105132.34 tokens/sec, time: 2m 59.4s
step 1200: train loss 4.0170, val loss 4.3883, speed: 105118.14 tokens/sec, time: 3m 7.2s
step 1250: train loss 3.9844, val loss 4.3670, speed: 105053.55 tokens/sec, time: 3m 15.1s
step 1300: train loss 3.9601, val loss 4.3501, speed: 105018.94 tokens/sec, time: 3m 23.0s
step 1350: train loss 3.9226, val loss 4.3310, speed: 105021.13 tokens/sec, time: 3m 30.8s
step 1400: train loss 3.9077, val loss 4.3136, speed: 105022.69 tokens/sec, time: 3m 38.6s
step 1450: train loss 3.8786, val loss 4.2988, speed: 104984.37 tokens/sec, time: 3m 46.4s
step 1500: train loss 3.8503, val loss 4.2784, speed: 104971.83 tokens/sec, time: 3m 54.3s
step 1550: train loss 3.8237, val loss 4.2614, speed: 105000.23 tokens/sec, time: 4m 2.0s
step 1600: train loss 3.8005, val loss 4.2503, speed: 104940.92 tokens/sec, time: 4m 10.0s
step 1650: train loss 3.7833, val loss 4.2264, speed: 104912.89 tokens/sec, time: 4m 17.8s
step 1700: train loss 3.7564, val loss 4.2210, speed: 104901.07 tokens/sec, time: 4m 25.7s
step 1750: train loss 3.7411, val loss 4.2056, speed: 104898.69 tokens/sec, time: 4m 33.5s
step 1800: train loss 3.7157, val loss 4.1930, speed: 104873.22 tokens/sec, time: 4m 41.4s
step 1850: train loss 3.7006, val loss 4.1794, speed: 104863.20 tokens/sec, time: 4m 49.2s
step 1900: train loss 3.6843, val loss 4.1722, speed: 104882.90 tokens/sec, time: 4m 57.0s
step 1950: train loss 3.6611, val loss 4.1588, speed: 104856.16 tokens/sec, time: 5m 4.8s
step 2000: train loss 3.6367, val loss 4.1533, speed: 104850.75 tokens/sec, time: 5m 12.7s
step 2050: train loss 3.6267, val loss 4.1479, speed: 104864.62 tokens/sec, time: 5m 20.4s
step 2100: train loss 3.6099, val loss 4.1354, speed: 104856.72 tokens/sec, time: 5m 28.3s
step 2150: train loss 3.5858, val loss 4.1333, speed: 104843.06 tokens/sec, time: 5m 36.1s
step 2200: train loss 3.5709, val loss 4.1218, speed: 104849.82 tokens/sec, time: 5m 43.9s
step 2250: train loss 3.5582, val loss 4.1131, speed: 104850.15 tokens/sec, time: 5m 51.7s
step 2300: train loss 3.5357, val loss 4.1120, speed: 104831.06 tokens/sec, time: 5m 59.6s
step 2350: train loss 3.5118, val loss 4.0939, speed: 104825.04 tokens/sec, time: 6m 7.5s
step 2400: train loss 3.5005, val loss 4.0901, speed: 104830.41 tokens/sec, time: 6m 15.3s
step 2450: train loss 3.4825, val loss 4.0816, speed: 104805.55 tokens/sec, time: 6m 23.2s
step 2500: train loss 3.4722, val loss 4.0818, speed: 104779.64 tokens/sec, time: 6m 31.1s
step 2550: train loss 3.4665, val loss 4.0707, speed: 104757.95 tokens/sec, time: 6m 39.0s
step 2600: train loss 3.4408, val loss 4.0571, speed: 104731.84 tokens/sec, time: 6m 46.9s
step 2650: train loss 3.4208, val loss 4.0688, speed: 104714.74 tokens/sec, time: 6m 54.8s
step 2700: train loss 3.4150, val loss 4.0522, speed: 104683.38 tokens/sec, time: 7m 2.7s
step 2750: train loss 3.3950, val loss 4.0587, speed: 104653.32 tokens/sec, time: 7m 10.7s
step 2800: train loss 3.3766, val loss 4.0551, speed: 104641.89 tokens/sec, time: 7m 18.6s
step 2850: train loss 3.3687, val loss 4.0322, speed: 104614.46 tokens/sec, time: 7m 26.5s
step 2900: train loss 3.3526, val loss 4.0499, speed: 104592.41 tokens/sec, time: 7m 34.4s
step 2950: train loss 3.3370, val loss 4.0578, speed: 104570.12 tokens/sec, time: 7m 42.4s
step 3000: train loss 3.3397, val loss 4.0406, speed: 104554.37 tokens/sec, time: 7m 50.3s
step 3050: train loss 3.3131, val loss 4.0303, speed: 104535.27 tokens/sec, time: 7m 58.2s
step 3100: train loss 3.3021, val loss 4.0228, speed: 104519.41 tokens/sec, time: 8m 6.1s
step 3150: train loss 3.2889, val loss 4.0201, speed: 104497.58 tokens/sec, time: 8m 14.0s
step 3200: train loss 3.2807, val loss 4.0175, speed: 104486.57 tokens/sec, time: 8m 21.9s
step 3250: train loss 3.2538, val loss 4.0151, speed: 104468.23 tokens/sec, time: 8m 29.9s
step 3300: train loss 3.2548, val loss 4.0097, speed: 104448.18 tokens/sec, time: 8m 37.8s
step 3350: train loss 3.2381, val loss 4.0185, speed: 104429.51 tokens/sec, time: 8m 45.7s
step 3400: train loss 3.2277, val loss 4.0254, speed: 104427.55 tokens/sec, time: 8m 53.6s
step 3450: train loss 3.2160, val loss 4.0147, speed: 104432.55 tokens/sec, time: 9m 1.4s
step 3500: train loss 3.2007, val loss 4.0030, speed: 104428.53 tokens/sec, time: 9m 9.3s
step 3550: train loss 3.1943, val loss 4.0054, speed: 104417.01 tokens/sec, time: 9m 17.2s
step 3600: train loss 3.1846, val loss 4.0042, speed: 104423.69 tokens/sec, time: 9m 25.0s
step 3650: train loss 3.1658, val loss 4.0055, speed: 104431.37 tokens/sec, time: 9m 32.8s
step 3700: train loss 3.1568, val loss 3.9942, speed: 104420.45 tokens/sec, time: 9m 40.7s
step 3750: train loss 3.1496, val loss 3.9841, speed: 104423.38 tokens/sec, time: 9m 48.5s
step 3800: train loss 3.1345, val loss 4.0056, speed: 104437.50 tokens/sec, time: 9m 56.3s
step 3850: train loss 3.1139, val loss 3.9984, speed: 104433.59 tokens/sec, time: 10m 4.2s
step 3900: train loss 3.1084, val loss 3.9971, speed: 104434.62 tokens/sec, time: 10m 12.0s
step 3950: train loss 3.0957, val loss 3.9887, speed: 104445.03 tokens/sec, time: 10m 19.8s
step 4000: train loss 3.0868, val loss 3.9860, speed: 104445.87 tokens/sec, time: 10m 27.6s
step 4050: train loss 3.0764, val loss 3.9925, speed: 104440.74 tokens/sec, time: 10m 35.5s
step 4100: train loss 3.0676, val loss 3.9859, speed: 104438.08 tokens/sec, time: 10m 43.4s
step 4150: train loss 3.0623, val loss 4.0036, speed: 104427.32 tokens/sec, time: 10m 51.3s
step 4200: train loss 3.0498, val loss 3.9939, speed: 104425.85 tokens/sec, time: 10m 59.1s
step 4250: train loss 3.0349, val loss 3.9905, speed: 104423.34 tokens/sec, time: 11m 7.0s
step 4300: train loss 3.0235, val loss 3.9972, speed: 104419.05 tokens/sec, time: 11m 14.9s
step 4350: train loss 3.0172, val loss 3.9864, speed: 104413.54 tokens/sec, time: 11m 22.7s
step 4400: train loss 2.9990, val loss 3.9889, speed: 104408.48 tokens/sec, time: 11m 30.6s
step 4450: train loss 2.9856, val loss 3.9947, speed: 104399.48 tokens/sec, time: 11m 38.5s
step 4500: train loss 2.9855, val loss 3.9878, speed: 104393.24 tokens/sec, time: 11m 46.4s
step 4550: train loss 2.9820, val loss 3.9869, speed: 104386.86 tokens/sec, time: 11m 54.3s
step 4600: train loss 2.9620, val loss 4.0056, speed: 104382.30 tokens/sec, time: 12m 2.2s
step 4650: train loss 2.9587, val loss 3.9961, speed: 104378.51 tokens/sec, time: 12m 10.1s
step 4700: train loss 2.9472, val loss 3.9785, speed: 104375.66 tokens/sec, time: 12m 17.9s
step 4750: train loss 2.9422, val loss 3.9992, speed: 104371.38 tokens/sec, time: 12m 25.8s
step 4800: train loss 2.9254, val loss 3.9912, speed: 104366.40 tokens/sec, time: 12m 33.7s
step 4850: train loss 2.9172, val loss 3.9920, speed: 104367.46 tokens/sec, time: 12m 41.5s
step 4900: train loss 2.9117, val loss 3.9990, speed: 104355.06 tokens/sec, time: 12m 49.5s
step 4950: train loss 2.8944, val loss 3.9963, speed: 104340.95 tokens/sec, time: 12m 57.4s
春江水似流。

临江仙 陈允平
红柳依然蝴蝶乱，汝江桥。
剪轻鸥荐日飞来。
叠叠匀酥相半掩，霏霏残梦归郎乱注疏篱。
属车忽到谢仙归。
毕竟归来如梦去，会容觞咏再来期。

一翦梅 陈允平
芍药闲来草碧初。

----------
往事，恨功名淡泪眼垂些。
幅酒难禁。
无缘一点恩光万红成。
而今宁许我堪歌更擘划，无计是愁人。

临江仙 魏了翁
思深契阔隐驹重，却倾不惜伤牵。
尊频劝客莫徘徊。
舣舟方把柂，更须取易相随。
无风吹我鬓毛
----------
模型已保存到model.pth
wandb: 🚀 View run absurd-frog-1 at: https://wandb.ai/simpxx-zhejiang-university/babygpt/runs/ysgr3tei
wandb: Find logs at: wandb/run-20250309_235239-ysgr3tei/logs
'''