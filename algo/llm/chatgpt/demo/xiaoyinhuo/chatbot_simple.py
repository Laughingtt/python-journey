import gradio as gr
import random
import time

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    with gr.Row():
        '''creates a new Textbox component, which is used to collect user input. 
        The show_label parameter is set to False to hide the label, 
        and the placeholder parameter is set'''
        query = gr.Textbox(show_label=False, placeholder="Enter text and press enter")

    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot], size='sm')
    a = chatbot.postprocess(chatbot.example_inputs())
    print(a)


    def respond(message, chat_history):
        bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history


    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
