import os
from langchain_community.graphs import Neo4jGraph


graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="test1234")


from langchain_experimental.graph_transformers import LLMGraphTransformer
from llm_qwen import CustomLLM

llm = CustomLLM(n=1)

llm_transformer = LLMGraphTransformer(llm=llm)

from langchain_core.documents import Document

text = """
Marie Curie, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris.
"""
documents = [Document(page_content=text)]
graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(f"Nodes:{graph_documents[0].nodes}")
print(f"Relationships:{graph_documents[0].relationships}")


graph.add_graph_documents(graph_documents)


# dbms.security.procedures.unrestricted=apoc.*
# apoc.import.file.enabled=true