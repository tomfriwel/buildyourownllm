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
        self.transition = [[0.0 for _ in range(vocab_size)] 
                          for _ in range(vocab_size)]
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, idx: List[List[int]]) -> List[List[List[float]]]:
        B = len(idx)  # batch size
        T = len(idx[0])  # sequence length
        
        logits = [
            [[0.0 for _ in range(self.vocab_size)] 
             for _ in range(T)]
            for _ in range(B)
        ]
        
        for b in range(B):
            for t in range(T):
                current_token = idx[b][t]
                logits[b][t] = self.transition[current_token]
                
        return logits

    def generate(self, idx: List[List[int]], max_new_tokens: int) -> List[int]:
        B = len(idx)
        T = len(idx[0])
        
        for _ in range(max_new_tokens):
            logits_batch = self(idx)
            for batch_idx, logits in enumerate(logits_batch):
                logits = logits[-1]
                total = sum(logits)
                logits = [logit / total for logit in logits]
                next_token = random.choices(
                    range(self.vocab_size),
                    weights=logits,
                    k=1
                )[0]
                idx[batch_idx].append(next_token)
        return idx
    
def get_batch(tokens, batch_size, block_size):
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

for iter in range(max_iters):
    x_batch, y_batch = get_batch(tokens, batch_size, block_size)
    for i in range(len(x_batch)):
        for j in range(len(x_batch[i])):
            x = x_batch[i][j]
            y = y_batch[i][j]
            model.transition[x][y] += 1

prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]

for tokens in model.generate(prompt_tokens, max_new_token):
    print(tokenizer.decode(tokens))
    print('-'*10)