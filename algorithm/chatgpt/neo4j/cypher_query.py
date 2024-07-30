from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

QUERY = """
"MATCH (m:Movie)-[:IN_GENRE]->(:Genre {name:$genre})
RETURN m.title, m.plot
ORDER BY m.imdbRating DESC LIMIT 5"
"""

graph.query(QUERY, genre="action")