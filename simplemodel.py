import random

random.seed(42) # 去掉此行，获得随机结果

prompt = "春江"
max_new_token = 100

with open('ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

transition = [[0 for _ in range(vocab_size)] for _ in range(vocab_size)]

for i in range(len(text) - 1):
    current_token_id = encode(text[i])[0]
    next_token_id = encode(text[i + 1])[0]
    transition[current_token_id][next_token_id] += 1

generated_token = encode(prompt)

for i in range(max_new_token - 1):
    current_token_id = generated_token[-1]
    logits = transition[current_token_id]
    total = max(sum(logits),1)
    logits = [logit / total for logit in logits]
    next_token_id = random.choices(range(vocab_size), weights=logits, k=1)[0]
    generated_token.append(next_token_id)
    current_token_id = next_token_id

print(decode(generated_token))