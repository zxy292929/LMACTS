import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain.schema import HumanMessage, SystemMessage
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["ZHIPUAI_API_KEY"] = "abac97da27af5add9e167776e9004986.7UIrxBHels5LncbL"
txts_folder = "txts"
txt_files = [f for f in os.listdir(txts_folder) if f.endswith('.xml')]

docs = []

for txt_file in txt_files:
    file_path = os.path.join(txts_folder, txt_file)
    loader = TextLoader(file_path)
    doc = loader.load()
    docs.extend(doc) 

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

class ConnectionRuleDesign:
    def ConnectionRuleDesign(dataset_name,unseen_dataset_desc):
        langchain_query = ChatZhipuAI(temperature=1, model='glm-4-plus')
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vector_store = Chroma(embedding_function=embeddings)
        max_batch_size = 41666
        for i in range(0, len(splits), max_batch_size):
            batch = splits[i:i + max_batch_size]
            _ = vector_store.add_documents(documents=batch)
        retriever = vector_store.as_retriever()
        RoleSetting="You are Quoc V. Le, a computer scientist and artificial intelligence researcher who is widely regarded as one of the leading experts in deep learning and neural network architecture search. Your work in"
        "this area has focused on developing efficient algorithms for searching the space of possible neural network architectures, with the goal of finding architectures that perform well on a given task "
        "while minimizing the computational cost of training and inference.\n"
        "The task at hand involves selecting the most appropriate connection rules for neural structure search in the field of spatiotemporal neural networks with reference to the external knowledge base, including designing the number of St-blocks, the connection mode, and the structure within each st-block. \n"
        "The following are the contents of the external knowledge base:\n"
        input="Now, you are tasked with Neural Architecture Search in the field of Spatiotemporal neural networks. Please take a deep breath and work on this problem step-by-step: based on the knowledge in the knowledge base and your previous experience, in order to achieve the best performance of the model, give specific recommendations on the following issues: 1. The number of St-blocks 2. How to connect St-blocks 3. Whether each ST-block is the same 4. The internal structure of each ST-block: (1) the number of nodes on which the operator can be placed and (2) how the nodes are connected. \n"
        "This is information for the unseen dataset:"
        "Your answer should be in the following format: For the unseen dataset, my suggestion is:\nReason:\n"
        retrieved_docs = vector_store.similarity_search(input)
        docs_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
        print("docs_content:",docs_content)
# messages = f"Question: {input}\n\nContext:\n{docs_content}"
        messages = [
            SystemMessage(content=RoleSetting+docs_content),
            HumanMessage(content=input)
        ]
        response = langchain_query.invoke(messages)
        print("response:",response.content)
        return response.content


