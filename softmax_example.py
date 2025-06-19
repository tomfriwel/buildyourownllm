import math
import torch
import torch.nn.functional as F

def normalize(prices):
    # Step 1: Calculate the total price
    total = sum(prices)
    
    # Step 2: Calculate normalized percentages
    percentages = [price / total for price in prices]
    
    return percentages

# Softmax calculation
def softmax(prices):
    logits = torch.tensor(prices, dtype=torch.float32)
    probs = F.softmax(logits, dim=-1)
    return probs.tolist()

# Example: Milk tea prices
prices = [20, 15, 10, 25, 30, 30, 31, 29, 5, 12, 18]  # Prices for various Milk Tea options

# Normalization
percentages = normalize(prices)

# Softmax
softmax_percentages = softmax(prices)

# Print results
print("Normalization Results:")
for i, percentage in enumerate(percentages):
    print(f"Milk Tea {i + 1}: {percentage * 100:.2f}%")

print("\nSoftmax Results:")
for i, percentage in enumerate(softmax_percentages):
    print(f"Milk Tea {i + 1}: {percentage * 100:.2f}%")