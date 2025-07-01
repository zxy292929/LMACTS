import os
from langchain.document_loaders import TextLoader
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
    loader = TextLoader("tt.txt")
    doc = loader.load()
    docs.extend(doc) 

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
langchain_query = ChatZhipuAI(temperature=1, model='glm-4-plus')
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(embedding_function=embeddings)
max_batch_size = 41666
for i in range(0, len(splits), max_batch_size):
    batch = splits[i:i + max_batch_size]
    _ = vector_store.add_documents(documents=batch)
retriever = vector_store.as_retriever()
input="We are going to do neural architecture search in the context of spatiotemporal data sets, please give me the ranking of operations that you think are best used to do neural architecture search."
retrieved_docs = vector_store.similarity_search(input)
docs_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
print("docs_content:",docs_content)
messages = f"Question: {input}\n\nContext:\n{docs_content}"
messages = [
    HumanMessage(content=input),
    SystemMessage(content=docs_content)
]

response = langchain_query.invoke(messages)
print("response:",response)


