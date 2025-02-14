import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model_checkpoint = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# model = model.to(device)
#
# sentence = '我叫张三，我住在苏州。'
#
# sentence_inputs = tokenizer(sentence, return_tensors="pt").to(device)
# sentence_generated_tokens = model.generate(
#     sentence_inputs["input_ids"],
#     attention_mask=sentence_inputs["attention_mask"],
#     max_length=128
# )
# sentence_decoded_pred = tokenizer.decode(sentence_generated_tokens[0], skip_special_tokens=True)
# print(sentence_decoded_pred)


sentences = ['我叫张三，我住在苏州。', '我在环境优美的苏州大学学习计算机。']

sentences_inputs = tokenizer(
    sentences,
    padding=True,
    max_length=128,
    truncation=True,
    return_tensors="pt"
).to(device)
sentences_generated_tokens = model.generate(
    sentences_inputs["input_ids"],
    attention_mask=sentences_inputs["attention_mask"],
    max_length=128
)
sentences_decoded_preds = tokenizer.batch_decode(sentences_generated_tokens, skip_special_tokens=True)
print(sentences_decoded_preds)