from langchain_community.llms import Ollama
llm = Ollama(model="gemma2")
llm.invoke("Why is the sky blue?")