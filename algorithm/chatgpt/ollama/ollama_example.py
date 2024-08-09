from openai import OpenAI



client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
)


chat_completion = client.chat.completions.create(
    messages=[
        {

            'role': 'user',
            'content': '你叫什么名字？',
        }
    ],
    model='qwen2:0.5b',
)

print(chat_completion)



