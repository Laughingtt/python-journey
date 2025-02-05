import torch
import torch.nn as nn

# 定义 Embedding 层
vocab_size = 10000  # 词汇表大小
hidden_size = 768   # Embedding 向量维度
token_embeddings = nn.Embedding(vocab_size, hidden_size)

# 定义输入
input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 形状为 [batch_size, seq_len] = [2, 3]

# 获取 Embedding 向量
embeddings = token_embeddings(input_ids)  # 形状为 [batch_size, seq_len, hidden_size] = [2, 3, 768]

print("Input IDs shape:", input_ids.shape)
print("Embeddings shape:", embeddings.shape)
print("Embeddings (first sample, first token):", embeddings[0, 0, :5])  # 打印第一个样本的第一个 token 的前5个维度