
from transformers import MarianMTModel, MarianTokenizer

# 加载翻译模型和 tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-zh'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# 输入待翻译的文本
text = "Hello, world!"

# 进行翻译
input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
outputs = model.generate(input_ids=input_ids)

# 解码并输出翻译结果
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translated_text)
