from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载 GPT-2 模型和 tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# # 修改模型结构或参数
# model.transformer.n_layers = 12 # 修改编码器层数为 12
# model.config.vocab_size = 10000 # 修改词表大小为 10000
#
# # 保存修改后的模型
# model.save_pretrained("modified_model")
#
# # 重新加载模型并生成
# model = GPT2LMHeadModel.from_pretrained("modified_model")
input_text = "hello"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(input_ids=input_ids, max_length=50, do_sample=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
