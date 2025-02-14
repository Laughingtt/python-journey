Tavily
# export TAVILY_API_KEY="..."
from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults()
search.invoke("what is the weather in SF")

"""
检索器
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()
retriever.get_relevant_documents("how to upload a dataset")[0]

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

"""
工具

"""
tools = [search, retriever_tool]

"""
代理
"""

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

from langchain import hub

# 获取要使用的提示 - 您可以修改这个！
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages

"""
创建代理

"""

from langchain.agents import create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == '__main__':
    agent_executor.invoke({"input": "hi!"})

    agent_executor.invoke({"input": "how can langsmith help with testing?"})

    agent_executor.invoke({"input": "whats the weather in sf?"})
