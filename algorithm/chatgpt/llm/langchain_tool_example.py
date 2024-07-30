from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool

"""
创建一个工具

"""


@tool
def multiply(first_int: int, second_int: int) -> int:
    """将两个整数相乘。"""
    return first_int * second_int


print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.invoke({"first_int": 4, "second_int": 5}))

"""
创建我们的提示信息

"""
from langchain.tools.render import render_text_description

rendered_tools = render_text_description([multiply])
print(rendered_tools)

from langchain_core.prompts import ChatPromptTemplate

system_prompt = f"""You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys."""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)
print("prompt: ", prompt)

"""
添加输出解析器
"""
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
from llm_qwen import CustomLLM

model = CustomLLM(n=1)
chain = prompt | model | JsonOutputParser()
res = chain.invoke({"input": "13乘以4是多少"})
print("res :", res)

"""
调用工具

"""
from operator import itemgetter

chain = prompt | model | JsonOutputParser() | itemgetter("arguments") | multiply
res2 = chain.invoke({"input": "13乘以4是多少"})
print("res2 :", res2)