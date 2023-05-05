
# curl https://api.openai.com/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer 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" \
#   -d '{
#     "model": "gpt-3.5-turbo",
#     "messages": [{"role": "user", "content": "大模型目前都有哪些!"}]
#   }'


import openai

# Set the API key
openai.api_key = "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"

# Define the model and prompt
model_engine = "text-davinci-003"
prompt = "我想学英语，你扮演英语老师的角色，我们进行情景对话，如果我表达的语法或者单词有问题，你可以直接指出来"
prompt2 = "任意主题都可以 "
prompt3 = "my name is tianjian , nice to meet you"

# Generate a response
completion = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

# Get the response text
message = completion.choices[0].text

print(message)


