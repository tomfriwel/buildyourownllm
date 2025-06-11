with open('ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

print(set(text))