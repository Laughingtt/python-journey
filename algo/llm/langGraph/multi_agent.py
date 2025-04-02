import langgraph
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# 初始化 OpenAI Chat 模型
llm = ChatOpenAI(api_key="",
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="qwen-turbo")


# 定义研究员节点，收集市场数据
def researcher_node(inputs):
    user_query = inputs["query"]
    response = llm([SystemMessage(content="你是市场研究员，收集关于" + user_query + "的最新市场数据。")])
    return {"research_data": response.content}


# 定义分析师节点，分析数据
def analyst_node(inputs):
    research_data = inputs["research_data"]
    response = llm([SystemMessage(content=f"你是数据分析师，请分析以下市场数据并提供洞察:\n{research_data}")])
    return {"analysis": response.content}


# 定义撰写者节点，生成报告
def writer_node(inputs):
    analysis = inputs["analysis"]
    response = llm([SystemMessage(content=f"你是商业撰写者，请根据以下分析撰写市场报告:\n{analysis}")])
    return {"report": response.content}


# 创建 LangGraph 有向图
graph = langgraph.Graph()
graph.add_node("researcher", researcher_node)
graph.add_node("analyst", analyst_node)
graph.add_node("writer", writer_node)

# 定义节点间连接
graph.add_edge("researcher", "analyst")
graph.add_edge("analyst", "writer")

# 设置初始节点
graph.set_entry_point("researcher")

# 编译运行图
workflow = graph.compile()

# 运行任务示例
user_input = "人工智能在医疗行业的市场趋势"
result = workflow.invoke({"query": user_input})

# 输出最终报告
print("市场分析报告:")
print(result["report"])
