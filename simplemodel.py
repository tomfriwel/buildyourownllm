import random

random.seed(42) # 随机种子，确保复现

prompt = "春江"
max_new_token = 100

with open('ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 词汇表
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#      a    b    c  ... (vocab_size)
#   a  0    1    0  ...
#   b  0    0    3  ...
#   c  4    0    0  ...
#  ... ... ... ... ... ... ...
# (vocab_size)
# vocab_size * vocab_size的二维数组，记录每个词的下一个词的出现次数
transition = [[0 for _ in range(vocab_size)] for _ in range(vocab_size)]

# 统计次数
for i in range(len(text) - 1):
    current_token_id = encode(text[i])[0]
    next_token_id = encode(text[i + 1])[0]
    transition[current_token_id][next_token_id] += 1

start_token_id = encode(prompt)[-1]
generated_token = [start_token_id]

for i in range(max_new_token - 1):
    current_token_id = generated_token[-1]
    logits = transition[current_token_id]
    total = sum(logits)
    # 归一化前 logits = [0, 10, ...., 298,..., 88, ..., 13, 0]
    #         len(logits) = vocab_size, sum(logits) = 6664
    # 归一化后 logits = [0/6664, 10/6664, ...., 298/6664,..., 88/6664, ..., 13/6664, 0/6664]
    #                = [0, 0.0015, ..., 0.0447, ..., 0.0132, ..., 0.00195, 0]
    logits = [logit / total for logit in logits]
    # 计算概率，随机采样，得到下一个词
    next_token_id = random.choices(range(vocab_size), weights=logits, k=1)[0]
    generated_token.append(next_token_id)
    current_token_id = next_token_id

print(decode(generated_token))