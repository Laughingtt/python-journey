import torch
import torch.nn as nn

# 定义输入
batch_size = 1
seq_len = 5
embed_dim = 768
head_dim = 768

# 定义 query
query = torch.randn(batch_size, seq_len, embed_dim)  # 形状为 [1, 5, 768]

# 定义线性变换层
self_q = nn.Linear(embed_dim, head_dim)  # 输入维度 768，输出维度 768

# 进行线性变换
output = self_q(query)  # 形状为 [1, 5, 768]

print("Query shape:", query.shape)
print("Query :", query)
print("Output shape:", output.shape)
print("Output:", output)