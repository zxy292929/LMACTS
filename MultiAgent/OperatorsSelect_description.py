import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain.schema import HumanMessage, SystemMessage
import re 
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["ZHIPUAI_API_KEY"] = "9ee64d5ef7e142fd8c94d0cf9f3ef3b9.KGeWPh8f9d8a2KIX"
# txts_folder = "txts"
# txt_files = [f for f in os.listdir(txts_folder) if f.endswith('.xml')]

# docs = []

# for txt_file in txt_files:
#     file_path = os.path.join(txts_folder, txt_file)
#     loader = TextLoader(file_path)
#     doc = loader.load()
#     docs.extend(doc) 

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)
class OperatorsSelect:
    def OperatorsSelect(unseen_dataset_desc):
        langchain_query = ChatZhipuAI(temperature=1, model='glm-4-plus')
        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        # vector_store = Chroma(embedding_function=embeddings)
        # max_batch_size = 41666
        # for i in range(0, len(splits), max_batch_size):
        #     batch = splits[i:i + max_batch_size]
        #     _ = vector_store.add_documents(documents=batch)
        # retriever = vector_store.as_retriever()
        RoleSetting='''You are Quoc V. Le, a computer scientist and artificial intelligence researcher who is widely regarded as one of the leading experts in deep 
        learning and 
        neural network architecture search. Your work in
        this area has focused on developing efficient algorithms for searching the space of possible neural network architectures, with the goal of finding 
        architectures that 
        perform well on a given task 
        while minimizing the computational cost of training and inference.\n
        The task at hand involves selecting the best operators for a specific dataset in the field of spatiotemporal neural networks according to its description 
        and characteristics. \n
        The following is the information for this unseen dataset:\n'''
        # input='''Now, you are tasked with Neural Architecture Search in the field of Spatiotemporal neural networks. Please take a deep breath and work on this problem step-by-step: 
        # based on your previous experience, considering the spatiotemporal domain and the characteristics of this dataset, rank the 20 operators you believe will bring the greatest 
        # performance improvements to the model.\n
        # Here are some examples of operator names: Skip Connection, Diffusion GCN, Transformer, Simplified Transformer, LSTM, GRU, RNN, 1D-CNN, Dynamic Convolutional Cell, N-Linear, 
        # Bottleneck, Convolutional Layer.\n
        # Your answer should be in the following format: For the unseen dataset: (1.[operator_name1] 2.[operator_name2] 3.[operator_name3] 4.[operator_name4] 5.[operator_name5] 
        # 6.[operator_name6] 7.[operator_name7] 8.[operator_name8] 9.[operator_name9] 10.[operator_name10] 11.[operator_name11] 12.[operator_name12] 13.[operator_name13] 
        # 14.[operator_name14] 15.[operator_name15] 16.[operator_name16] 17.[operator_name17] 18.[operator_name18] 19.[operator_name19] 20.[operator_name20])\nReason:\n'''
        input='''Now, you are tasked with Neural Architecture Search in the field of Spatiotemporal neural networks. Please take a deep breath and work on this problem step-by-step: 
        based on your previous experience and the introduction of the operators below, please rank the following 11 operators by the performance improvement they are likely to bring
          to the model.\n
        1. 'skip_connect': Skip Connection  Layer, a mechanism to bypass intermediate layers, preserving original input information and supporting residual learning. \n
        2. 'gated_tcn': The Gated TCN (Gated Temporal Convolutional Network) Layer, consists of two parallel temporal convolution layers, TCN-a and TCN-b. These layers process the 
        input data using the tanh and sigmoid activation functions, respectively, to extract temporal features. The processed outputs are fused through a multiplication operation, 
        and this gating mechanism allows the model to dynamically adjust the flow of information, enhancing its ability to capture important temporal features. Additionally, the 
        output of the Gated TCN module is added to the original input via a skip connection, forming a residual connection. This helps alleviate the vanishing gradient problem in 
        deep network training and enables the network to more effectively learn long-term dependencies. \n
        3. 'diff_gcn': Diffusion Graph Convolutional Network Layer, extends graph convolution by modeling the diffusion process of node signals over multiple steps.  It aggregates 
        information from a node's neighbors and higher-order neighbors, capturing long-range dependencies and temporal dynamics for tasks like spatial-temporal forecasting. \n
        4. 'trans': Transformer Layer, uses self-attention to model relationships between stages, effective for capturing global interactions. \n
        5. 's_trans': Spatial Transformer Layer, combines spatial attention and convolutional operations to process input data with a spatial-temporal structure.  It first applies 
        a spatial attention mechanism to capture long-range dependencies between nodes, using a modified attention layer (SpatialAttentionLayer).  Then, the layer performs two 1D 
        convolution operations to transform the feature dimensions.  Layer normalization and dropout are applied to stabilize training and prevent overfitting.  Finally, the output
          is reshaped and permuted to match the required format for further processing.  This layer is designed to handle high-dimensional time-series or graph-structured data, 
          enabling the model to focus on important spatial features while learning efficient representations. \n
        6. 'lstm': Long Short-Term Memory Layer, captures long-term dependencies in sequences using gated mechanisms. \n
        7. 'gru': Gated Recurrent Unit Layer, a simplified LSTM variant, balances performance and computational cost. \n
        8. 'nlinear': NLinear Layer is a variant of the LTSF-Linear model designed to handle distribution shifts in time series data. It works by first subtracting the last value 
        of the input sequence, effectively normalizing the sequence. Then, the normalized data passes through a linear layer to learn the underlying patterns. Finally, the subtracted part is added back to the output, producing the final prediction. This subtraction and addition process helps to stabilize the model when there are shifts in the distribution of the dataset, making it more robust for tasks involving time series with varying trends or seasonal patterns. \n
        9. 'conv': Convolutional Layer, extracts local features from input data, crucial for tasks like image and graph analysis. \n
        10. 'rnn': Recurrent Neural Network Layer, processes sequential data by maintaining a hidden state, capturing temporal dependencies efficiently. \n
        11. '1d-cnn': One-Dimensional Convolutional Neural Network Layer, captures local patterns in sequential data, commonly used for time series and text analysis. \n
        Your answer should be in the following format: For the unseen dataset: (1.[operator_name1] 2.[operator_name2] 3.[operator_name3] 4.[operator_name4] 5.[operator_name5] 
        6.[operator_name6] 7.[operator_name7] 8.[operator_name8] 9.[operator_name9] 10.[operator_name10] 11.[operator_name11])\nReason:\n'''

        # retrieved_docs = vector_store.similarity_search(RoleSetting+input)
        # docs_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
        # print("docs_content:",docs_content)
        # messages = f"Question: {input}\n\nContext:\n{docs_content}"
        messages = [
            SystemMessage(content=RoleSetting+unseen_dataset_desc),
            HumanMessage(content=input)
        ]
        response = langchain_query.invoke(messages)   
        print("response.content:", response.content)
        extracted_text = re.search(r"For the unseen dataset:\s*(.*?)\s*Reason:", response.content, re.S)
        result = extracted_text.group(1).strip() if extracted_text else None
        operators = re.findall(r"\d+\.\s(.+?)(?:\s*\(.*?\))?\s*(?=\d+\.|\))", result, re.S)
        return operators
def main():
    # unseen_dataset_desc = '''unseen_dataset_desc: Dataset Type: Unseen\n
    # Temporal Features:\n
    # - Temporal Granularity: 0.0417\n
    # - Time Span: 726\n
    # - Cyclic Patterns: Daily\n
    # - Cycle Length: 24\n 
    # Statistical Features:\n
    # - Sum Values: 54356.742980\n
    # - Mean Change: 0.000107\n
    # - Mean Second Derivative Central: 0.000005\n
    # - Median: 3.392667\n
    # - Mean: 3.120364\n
    # - Length: 17420.0\n
    # - Standard Deviation: 2.509342\n
    # - Variation Coefficient: 0.804182\n
    # - Variance: 6.296798\n
    # - Skewness: -0.830492\n
    # - Kurtosis: 0.849066\n
    # - Root Mean Square: 4.004182\n
    # - Absolute Sum Of Changes: 12184.568167\n
    # - Longest Strike Below Mean: 73.0\n'''

    unseen_dataset_desc = '''unseen_dataset_desc: Dataset Type: Unseen\n
Temporal Features:\n
- Temporal Granularity: 1\n
- Time Span: 9862\n
- Cyclic Patterns: -\n
- Cycle Length: -\n\n

Statistical Features:\n
- Sum Values: 5298.189\n
- Mean Change: -0.000003371029\n
- Mean Second Derivative Central: 0.00000004662725\n
- Median: 0.6952419\n
- Mean: 0.6982326\n
- Length: 7588.0\n
- Standard Deviation: 0.08045370\n
- Variation Coefficient: 0.1152248\n
- Variance: 0.006472797\n
- Skewness: -0.1517160\n
- Kurtosis: -0.6799092\n
- Root Mean Square: 0.7028524\n
- Absolute Sum Of Changes: 16.68794\n
- Longest Strike Below Mean: 2762.0\n'''
    
    result = OperatorsSelect.OperatorsSelect(unseen_dataset_desc)

# Run the main function
if __name__ == "__main__":
    main()