import requests
import json

url = "http://localhost:11434/v1/chat/completions"

payload = json.dumps({
  "model": "qwen2:0.5b",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "你是谁，能帮我介绍一下关羽吗？!"
    }
  ]
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
