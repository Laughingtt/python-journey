from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(api_key="",
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="qwen-turbo")

messages = [
    SystemMessage(content="你是一个智能助手，回答问题要简洁明了。"),
    HumanMessage(content="你好！，你是谁")
]

# 调用模型生成回复
response = llm(messages)

# 输出回复
print(response.content)