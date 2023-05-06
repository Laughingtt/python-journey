from transformers import MarianMTModel, MarianTokenizer

# 加载翻译模型和 tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-zh'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# 修改模型结构或参数
model.encoder.layers = model.encoder.layers[:-1] # 删除最后一层编码器
model.config.d_model = 256 # 修改隐藏层大小为 256

# 保存修改后的模型
model.save_pretrained('modified_model')

# 重新加载模型并进行翻译
model = MarianMTModel.from_pretrained('modified_model')
input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
outputs = model.generate(input_ids=input_ids)
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translated_text)
