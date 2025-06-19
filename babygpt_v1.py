import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
import time

torch.manual_seed(42)

prompts = ["春江", "往事"] # 推理的输入prompts，模型将基于这些提示生成文本
max_new_token = 100 # 推理生成的最大tokens数量，控制生成文本的长度

max_iters = 5000 # 训练的最大迭代次数，决定训练的总轮数
eval_iters = 100 # 评估的迭代次数，用于计算模型在验证集上的损失
eval_interval = 200 # 评估的间隔，每隔多少次训练迭代进行一次评估
batch_size = 32 # 每个批次的大小，决定每次训练使用的数据量
block_size = 8 # 每个序列的最大长度，限制输入序列的长度
learning_rate = 1e-2 # 学习率，控制参数更新的步伐
n_embed = 32 # 嵌入层的维度，决定每个token的向量表示的大小
tain_data_ratio = 0.9 # 训练数据占数据集的比例，剩下的是验证数据

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu' # 选择设备，优先使用GPU或MPS，如果不可用则使用CPU

with open('ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

class Tokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text))) # 想象成一个字母表，包含了所有独特的字符
        self.vocab_size = len(self.chars) # 字母表的大小
        self.stoi = {ch: i for i, ch in enumerate(self.chars)} # 字符到索引的映射，比如字母表中的"a"对应索引0
        self.itos = {i: ch for i, ch in enumerate(self.chars)} # 索引到字符的映射，比如索引0对应字母表中的"a"

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s] # 把字符串转换成索引列表，就像把单词拆成字母后查字母表的索引

    def decode(self, l: List[int]) -> str:
        return ''.join([self.itos[i] for i in l]) # 把索引列表转换回字符串，就像根据索引重新拼出单词

class BabyGPT(nn.Module):

    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()

        # 每个num_embeddings对应一个长度为embedding_dim的向量
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=n_embd
        ) # 嵌入层，把token映射到n_embd维空间，就像把字母表中的字母映射到一个多维空间中的点

        self.lm_head = nn.Linear(
            in_features=n_embd,
            out_features=vocab_size
        ) # 线性层，把n_embd维空间映射到vocab_size维空间，就像从多维空间回到字母表

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx) # 获得token的嵌入表示 (B,T,n_embd)，就像把句子中的每个字母映射到多维空间中的点
        logits = self.lm_head(tok_emb) # 通过线性层，把embedding结果重新映射回vocab_size维空间 (B,T,vocab_size)，就像预测每个字母的下一个可能的字母

        if targets is None: # 推理场景，不需要计算损失值
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # 把(B,T,C)的形状转换为(B*T,C)，就像把二维表格拉平成一维列表
            targets = targets.view(B*T) # 把(B,T)的形状转换为(B*T)，同样是为了适配损失函数的输入
            loss = F.cross_entropy(logits, targets) # 计算交叉熵损失，就像比较预测的字母和实际的字母是否一致
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx) # logits的形状是(B,T,vocab_size)，每一个token都计算了下一个token的概率
            logits = logits[:, -1, :] # 实际上我们只需要最后一个token算出来的值，就像只关心句子最后一个字母的预测
            probs = F.softmax(logits, dim=-1) # 使用softmax函数算概率分布，这里dim=-1表示对最后一个维度进行softmax，就像把预测的可能性归一化为百分比
            idx_next = torch.multinomial(probs, num_samples=1) # 根据概率分布随机采样，这里num_samples=1表示采样一个token，就像根据概率掷骰子选下一个字母
            idx = torch.cat((idx, idx_next), dim=1) # 把采样的token拼接到序列后面，就像把新选的字母加到句子后面
        return idx

tokenizer = Tokenizer(text) # 初始化分词器，就像准备好一本字典
vocab_size = tokenizer.vocab_size # 获取词汇表大小，就像知道字典里有多少个字

raw_data = torch.tensor(tokenizer.encode(text), dtype=torch.long).to(device) # 把文本编码成数字序列，就像把句子翻译成数字语言
n = int(tain_data_ratio*len(raw_data)) # 计算训练数据的分割点，就像决定训练和验证的比例

data = {'train': raw_data[:n], 'val': raw_data[n:]} # 分割数据集，就像把字典分成学习部分和测试部分

def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 随机选择起始点，就像从书中随机抽取一段文字
    x = torch.stack([data[i:i+block_size] for i in ix]) # 构造输入序列，就像选取连续的文字片段
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # 构造目标序列，就像选取输入序列的下一个字符
    x, y = x.to(device), y.to(device) # 把数据移动到设备上，就像把书放到桌子上准备阅读
    return x, y

@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, eval_iters):
    '''
    计算模型在训练集和验证集上的损失
    '''
    out = {}
    model.eval() # 切换到评估模式，就像学生在考试时不允许修改答案
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # 初始化损失值，就像准备一个空的成绩单
        for k in range(eval_iters):
            x, y = get_batch(data[split], batch_size, block_size) # 获取一个批次的数据，就像从训练集或验证集中抽取一组句子
            _, loss = model(x, y) # 计算损失值，就像评估学生的表现
            losses[k] = loss.item() # 记录损失值，就像把成绩写到成绩单上
        out[split] = losses.mean() # 计算平均损失值，就像计算平均成绩
    model.train() # 切换回训练模式，就像学生回到课堂继续学习
    return out

model = BabyGPT(vocab_size, n_embed).to(device)

# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # 优化器，就像教练根据表现调整策略

start_time = time.time()
tokens_processed = 0

for iter in range(max_iters):
    x, y = get_batch(data['train'], batch_size, block_size) # 获取一个批次的数据，就像从训练集中抽取一组句子
    logits, loss = model(x, y) # 前向传播，计算预测值和损失，就像学生根据教练的策略完成任务并计算表现
    optimizer.zero_grad(set_to_none=True) # 清空梯度，就像擦掉之前的笔记
    loss.backward() # 反向传播，计算梯度，就像教练根据表现给出改进建议
    optimizer.step() # 更新参数，就像学生根据建议调整策略

    tokens_processed += batch_size * block_size # 记录处理的token数量

    if iter % eval_interval == 0:
        elapsed = time.time() - start_time
        tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0 # 计算处理速度
        losses = estimate_loss(model, data, batch_size, block_size, eval_iters) # 评估模型在训练集和验证集上的表现
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, speed: {tokens_per_sec:.2f} tokens/sec")

# 推理
prompt_tokens = torch.stack([torch.tensor(tokenizer.encode(p)).to(device) for p in prompts]) # 把输入的提示词编码成token，就像把句子拆成字母

# 生成
result = model.generate(prompt_tokens, max_new_token) # 根据提示词生成新的token序列，就像根据开头的几个字母续写句子

# 解码并打印结果
for tokens in result:
    print(tokenizer.decode(tokens.tolist())) # 把生成的token解码成字符串，就像把字母拼成句子
    print('-'*10)