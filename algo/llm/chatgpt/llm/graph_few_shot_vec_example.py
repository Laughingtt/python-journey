from langchain_community.embeddings import HuggingFaceEmbeddings

examples = [
    {
        "question": "有多少位艺术家？",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie) RETURN count(DISTINCT a)",
    },
    {
        "question": "哪些演员在电影《Casino》中出演？",
        "query": "MATCH (m:Movie {{title: 'Casino'}})<-[:ACTED_IN]-(a) RETURN a.name",
    },
    {
        "question": "汤姆·汉克斯参演了多少部电影？",
        "query": "MATCH (a:Person {{name: 'Tom Hanks'}})-[:ACTED_IN]->(m:Movie) RETURN count(m)",
    },
    {
        "question": "列出电影《辛德勒的名单》的所有流派",
        "query": "MATCH (m:Movie {{title: 'Schindler\\'s List'}})-[:IN_GENRE]->(g:Genre) RETURN g.name",
    },
    {
        "question": "哪些演员曾在喜剧和动作两种类型的电影中工作过？",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genre), (a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genre) WHERE g1.name = 'Comedy' AND g2.name = 'Action' RETURN DISTINCT a.name",
    },
    {
        "question": "哪些导演曾和至少三位名为“约翰”的演员合作过？",
        "query": "MATCH (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Person) WHERE a.name STARTS WITH 'John' WITH d, COUNT(DISTINCT a) AS JohnsCount WHERE JohnsCount >= 3 RETURN d.name",
    },
    {
        "question": "识别导演也在电影中扮演了角色的电影。",
        "query": "MATCH (p:Person)-[:DIRECTED]->(m:Movie), (p)-[:ACTED_IN]->(m) RETURN m.title, p.name",
    },
    {
        "question": "找出数据库中拥有最多电影的演员。",
        "query": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a.name, COUNT(m) AS movieCount ORDER BY movieCount DESC LIMIT 1",
    },
]

from langchain_community.vectorstores import Neo4jVector, FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True}

embed_model = HuggingFaceEmbeddings(
    model_name='D:\\model\\m3e-base',
    model_kwargs={'device': 'cpu'},
    encode_kwargs=embedding_encode_kwargs
)

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embed_model,
    FAISS,  # Neo4jVector,
    k=5,
    input_keys=["question"],
)

example_prompt = PromptTemplate.from_template(
    "用户输入：{question}\nCypher查询：{query}"
)

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="你是一个Neo4j专家。给定一个输入问题，创建一个语法正确的Cypher查询来运行。\n\n下面是模式信息\n{schema}.\n\n以下是一些问题及其相应的Cypher查询的示例。",
    suffix="用户输入：{question}\nCypher查询：",
    input_variables=["question", "schema"],
)

print(prompt.format(question="有多少位艺术家？", schema="foo"))

# Neo4j
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from llm_qwen import CustomLLM

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="test1234")
llm = CustomLLM(n=1)

chain = GraphCypherQAChain.from_llm(
    graph=graph, llm=llm, cypher_prompt=prompt, verbose=True
)

chain.invoke("数据库中电影数量有多少？")
