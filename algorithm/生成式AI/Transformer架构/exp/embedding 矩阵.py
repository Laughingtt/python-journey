"""

向词表中添加新 token 后，必须重置模型 embedding 矩阵的大小，也就是向矩阵中添加新 token 对应的 embedding，这样模型才可以正常工作，将 token 映射到对应的 embedding。

调整 embedding 矩阵通过 resize_token_embeddings() 函数来实现，例如对于前面的例子：
"""

from transformers import AutoTokenizer, AutoModel

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

print('vocabulary size:', len(tokenizer))
num_added_toks = tokenizer.add_tokens(['[ENT_START]', '[ENT_END]'], special_tokens=True)
print("After we add", num_added_toks, "tokens")
print('vocabulary size:', len(tokenizer))

model.resize_token_embeddings(len(tokenizer))
print(model.embeddings.word_embeddings.weight.size())

# Randomly generated matrix
print(model.embeddings.word_embeddings.weight[-2:, :])