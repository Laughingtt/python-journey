import json, uuid, requests


def requests_chat_completions_deepseek(query):
    url = "https://shdata-aiweb-pre.shdatacloud.com/api/v1/chat/completions"
    headers = {'Content-Type': 'application/json',
               'Authorization': 'Bearer'  # apikey粘贴在这里
               }
    query_json = {
        "chatId": str(uuid.uuid4()),  # 每次调用生成新的随机chatid， 多轮对话需要保持chatid一致
        "stream": False,  # 是否要流式输出
        "variables": {},  # 暂时为空dict就行
        "messages": [
            {"role": "user", "content": query}  # 请求放在这里，模式同通用openai的调用方式
        ],
    }
    r = requests.post(url, headers=headers, data=json.dumps(query_json), timeout=120)
    print(r.json())


#     return r.json()['choices'][0]['message']['content']


query = "你是谁"
xx = requests_chat_completions_deepseek(query)
