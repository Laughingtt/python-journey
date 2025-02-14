# pip install --upgrade --quiet  langchain
# pip install --upgrade --quiet  langchain-openai

from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from llm_qwen import CustomLLM

llm = CustomLLM(n=1)

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="test1234")

# Insert some movie data
graph.query(
    """
MERGE (m:Movie {title:"Top Gun"})
WITH m
UNWIND ["Tom Cruise", "Val Kilmer", "Anthony Edwards", "Meg Ryan"] AS actor
MERGE (a:Actor {name:actor})
MERGE (a)-[:ACTED_IN]->(m)
"""
)

chain = GraphCypherQAChain.from_llm(
    llm, graph=graph, verbose=True
)

chain.run("Who played in Top Gun?")