import gradio as gr
import json
import requests
import time

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


def user_action(user_msg, bot):
    bot.append([user_msg, None])
    return '', bot

def bot_action(bot):
    user_msg = bot[-1][0]
    current_line = {'role': 'user', 'content': user_msg}
    if len(bot) == 1:
        prompt = [current_line]
    else:
        prompt = []
        for m in bot[:-1]:
            prompt.append({'role': 'user', 'content': m[0]})
            prompt.append({'role': 'assistant', 'content': m[1]})
        prompt.append(current_line)

    response = call_model(prompt)
    bot[-1][1] =  ''

    for c in response:
        bot[-1][1] += c
        time.sleep(0.005)
        yield bot
    
    return bot
        
   
def vote(data: gr.LikeData):
    return None


with gr.Blocks() as demo:
    
    with gr.Row():
        gr.Image('c:/dc/wm.png', width=100, scale=0)            
        gr.Markdown(
        '''
        # 知心大姐“唠五毛”
        
        智能小助手：小萤火，小名叫萤萤，希望微微萤火能照亮诉说者前行的路和心灵的光...
        
        一个懂你的陪伴型机器人，为你打造一片心灵的栖息地。
        在这里，你可以尽情倾诉，释放内心的情感，让心灵得到慰藉。让我们开始今天的谈话吧！
        ''')
    
    with gr.Row():

        with gr.Column(scale=3):
            bot = gr.Chatbot(
                bubble_full_width=False,
                avatar_images=('c:/dc/xxx.jpg', 'c:/dc/zgp.jpg')
                )

            
            with gr.Row():
                user_msg = gr.Textbox(
                    show_label=False,
                    placeholder='you can ask anything！')
                clear_btn = gr.ClearButton([user_msg, bot],scale=0, value='清除对话')
                
            user_msg.submit(user_action, [user_msg, bot], [user_msg, bot], queue=False).then(
                    bot_action, bot, bot)
        
            bot.like(vote, None, None)

        with gr.Column(scale=1):
            
            radio = gr.Radio(
                ['我是讨好型人格，感觉自己活的很卑微不快乐，我可以改变吗？',
                 '失恋为什么这么痛苦，能从心理学的角度帮我分析一下吗？',
                 '抑郁症怎么解决呢？',
                 '如果我不够优秀的话，是不是在别人眼里就没有价值？'],
                label='试试这些问题吧...')
            
            btn = gr.Button(value='提交一下试试看')
 
            def run_sample(txt):
                return txt, None
            
            btn.click(run_sample, radio, outputs=[user_msg, bot], queue=False).then(
                user_action, [user_msg, bot], [user_msg, bot], queue=False).then(
                    bot_action, bot, bot)

demo.launch()