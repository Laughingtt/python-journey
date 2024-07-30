def load_chain():
    # 切分文件
    # 目标文件夹
    tar_dir = ["/mnt/workspace/pre_knowledge_db"]

    # 加载目标文件
    docs = []
    for dir_path in tar_dir:
        docs.extend(get_text(dir_path))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)

    # 加载问答链
    # init embedding model
    embedding_model_name = '/mnt/workspace/models/bce-embedding-base_v1'
    embedding_model_kwargs = {'device': 'cuda:0'}
    embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True}

    embed_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs
    )
    # 加载重排序模型
    reranker_args = {'model': '/mnt/workspace/models/bce-reranker-base_v1', 'top_n': 3, 'device': 'cuda:0'}
    reranker = BCERerank(**reranker_args)
    # embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")

    # example 1. retrieval with embedding and reranker
    retriever = FAISS.from_documents(split_docs, embed_model,
                                     distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT).as_retriever(
        search_type="similarity", search_kwargs={"score_threshold": 0.3, "k": 10})

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=retriever
    )

    # 加载自定义 LLM
    llm = InternLM_LLM(model_path="/mnt/workspace/.cache/modelscope/Shanghai_AI_Laboratory/internlm2-chat-7b")
    # llm = InternLM_LLM(model_path="/mnt/workspace/models/merged")
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    # conversation_llm = ConversationChain(
    #     llm=llm,
    #     verbose=True,
    #     memory=ConversationBufferMemory()
    # )

    # 定义一个 Prompt Template
    template = """请参考以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。尽量用温和的语气鼓励，安慰提问者。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    # 运行 chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever, return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    return qa_chain