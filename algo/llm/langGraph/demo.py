from langgraph.graph import StateGraph


# 定义状态
class ChatState:
    def __init__(self, messages=None):
        self.messages = messages or []


# 创建状态机
workflow = StateGraph(ChatState)

def agent_one(state):
    state.messages.append("Agent One: Checking data...")
    return {"state": state}  # 返回字典

def agent_two(state):
    state.messages.append("Agent Two: Analyzing request...")
    return {"state": state}  # 返回字典

workflow.add_node("agent_one", agent_one)
workflow.add_node("agent_two", agent_two)

# 定义结束节点
def end(state):
    return {"state": state}  # 返回字典

# 添加结束节点到图中
workflow.add_node("end", end)

# 让 agent_one 执行完后，进入 agent_two
workflow.set_entry_point("agent_one")
workflow.add_edge("agent_one", "agent_two")
workflow.add_edge("agent_two", "end")
workflow.set_finish_point("end")

# 重新编译
app = workflow.compile()

# 运行状态机
result = app.invoke({"state": ChatState()})  # 传入字典


# 输出结果
print(result["state"].messages)
# 输出:
# ['Agent One: Checking data...', 'Agent Two: Analyzing request...']
