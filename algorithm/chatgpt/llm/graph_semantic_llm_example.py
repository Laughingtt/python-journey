from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

# 导入通用所需的内容
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool

"""
实现的函数是用于检索有关电影或演员的信息。
"""
description_query = """
MATCH (m:Movie|Person)
WHERE m.title CONTAINS $candidate OR m.name CONTAINS $candidate
MATCH (m)-[r:ACTED_IN|HAS_GENRE]-(t)
WITH m, type(r) AS type, collect(COALESCE(t.name, t.title)) AS names
WITH m, type + ": " + REDUCE(s = "", n IN names | s + n + ", ") AS types
WITH m, COLLECT(types) AS contexts
WITH m, "type:" + LABELS(m)[0] + "\ntitle: " + COALESCE(m.title, m.name) 
       + "\nyear: " + COALESCE(m.released, "") + "\n" +
       REDUCE(s = "", c IN contexts | s + SUBSTRING(c, 0, SIZE(c) - 2) + "\n") AS context
RETURN context LIMIT 1
"""

from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from llm_qwen import CustomLLM

llm = CustomLLM(n=1)

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="test1234")

def get_information(entity: str) -> str:
    try:
        data = graph.query(description_query, params={"candidate": entity})
        return data[0]["context"]
    except IndexError:
        return "未找到相关信息"

"""
使用LLM代理仅填充输入参数来使用工具。为了向LLM代理提供有关何时使用工具及其输入参数的附加信息，我们将该函数包装为一个工具。
"""

from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool


class InformationInput(BaseModel):
    entity: str = Field(description="在问题中提到的电影或人物")


class InformationTool(BaseTool):
    name = "Information"
    description = (
        "当您需要回答有关各种演员或电影的问题时很有用"
    )
    args_schema: Type[BaseModel] = InformationInput

    def _run(
            self,
            entity: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """使用该工具。"""
        return get_information(entity)

    async def _arun(
            self,
            entity: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """异步使用该工具。"""
        return get_information(entity)

"""
LangChain表达语言非常方便地定义与语义层上的图数据库进行交互的代理。

"""

from typing import List, Tuple

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [InformationTool()]

llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])


from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent


agent = create_tool_calling_agent(llm, tools, prompt)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "Who played in Casino?"})

