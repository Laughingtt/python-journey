import gradio as gr
import random
import os
import json
import requests

def call_model(prompt):
    token = ''
    url = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-lite-8k'
    url += '?access_token=' + token

    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({
        "messages": prompt,
        "disable_search": False,
        "enable_citation": False
        # "max_output_tokens": 500
    })

    resp = requests.request("POST", url, headers=headers, data=payload)
    result = json.loads(resp.text)['result']

    return result


def chat(user_input, history=[]):
    current_line = {'role': 'user', 'content': user_input}
    if len(history) == 0:
        prompt = [current_line]
        history = [current_line]
    else:
        prompt = history.append(current_line)

    response = call_model(prompt)
    history.append({'role': 'assistant', 'content': response})
    return response, history


if __name__ == '__main__':
    response, history = chat("inhao",[])
    print(response)
    print(history)