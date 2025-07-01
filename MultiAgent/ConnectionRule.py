import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain.schema import HumanMessage, SystemMessage
from nas_bench_graph.readbench import light_read, read
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import re 
import itertools
from nas_bench_graph.architecture import Arch
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["ZHIPUAI_API_KEY"] = "7fbfecceaa814ec28a5179b23b2c111b.IGhdADFnYOfBYSgp"
class ConnectionRule:

    @classmethod
    def ConnectionRule(cls,unseen_dataset_desc,similarity_scores):
        langchain_query = ChatZhipuAI(temperature=0.3, model='glm-4-plus')
        lines=[]
        absolute_path="/root/DesiGNN-copy7/datasets/Benchmark datasets/"
        length = len(next(iter(similarity_scores.values())))
        for i in range(length):
            path=absolute_path+str(list(next(iter(similarity_scores.values())).keys())[i])+"/architecture_ranking.file"
            with open(path, "r") as f:
                lines.append([next(f) for _ in range(6)]) 
        RoleSetting="You are a machine learning expert proficient in spatiotemporal neural network design and macro topological architecture selection. \n"
        input=f'''Your task is to select the macro topological architectures for the unseen dataset based on the incomplete statistical ranking of the macro topological architectures from similar datasets, leading to a low MAE architecture search space.
The following are some features of the unseen datasets:{unseen_dataset_desc}\n
In the context of spatiotemporal neural networks, the design of a model is described by two main components: the operation list and the macro topological architecture.Here are the detailed settings:
The 6 available operations are as follows:
- 'skip_connect': Skip Connection, a mechanism to bypass intermediate layers, preserving original input information and supporting residual learning. 
- 'gated_tcn': The Gated TCN (Gated Temporal Convolutional Network) Layer, consists of two parallel temporal convolution layers, TCN-a and TCN-b. These layers process the input data using the tanh and sigmoid activation functions, respectively, to extract temporal features. The processed outputs are fused through a multiplication operation, and this gating mechanism allows the model to dynamically adjust the flow of information, enhancing its ability to capture important temporal features. Additionally, the output of the Gated TCN module is added to the original input via a skip connection, forming a residual connection. This helps alleviate the vanishing gradient problem in deep network training and enables the network to more effectively learn long-term dependencies. 
- 'diff_gcn': Diffusion Graph Convolutional Network Layer, extends graph convolution by modeling the diffusion process of node signals over multiple steps.  It aggregates information from a node's neighbors and higher-order neighbors, capturing long-range dependencies and temporal dynamics for tasks like spatial-temporal forecasting.
- 'trans': Transformer, uses self-attention to model relationships between stages, effective for capturing global interactions.
- 's_trans': Spatial Transformer Layer, combines spatial attention and convolutional operations to process input data with a spatial-temporal structure.  It first applies a spatial attention mechanism to capture long-range dependencies between nodes, using a modified attention layer (SpatialAttentionLayer).  Then, the layer performs two 1D convolution operations to transform the feature dimensions.  Layer normalization and dropout are applied to stabilize training and prevent overfitting.  Finally, the output is reshaped and permuted to match the required format for further processing.  This layer is designed to handle high-dimensional time-series or graph-structured data, enabling the model to focus on important spatial features while learning efficient representations. 
- 'none': This operation indicates that there is no operational connection between the two stages, it breaks the gradient flow between two stages.
The macro topological architecture design is as follows:
The neural network block is defined by 10 operations (i.e., op_list = [op0, op1, op2, op3, op4, op5, op6, op7, op8, op9]), which represent the operations executed between various stages of the block.  This block comprises 5 stages, labeled as s0, s1, s2, s3 and s4, each corresponding to distinct feature maps in the neural network.
s0 serves as the input feature map for this block.
s1 will be calculated by s1 = op0(s0).
s2 will be calculated by s2 = op1(s0) + op2(s1).
s3 will be calculated by s3 = op3(s0) + op4(s1) + op5(s2).
s4 will be calculated by s4 = op6(s0) + op7(s1) + op8(s2) + op9(s3).
Note that s4 becomes the output for this block and serves as the input for the subsequent block.
Then the implementation of the block will be:
class Block(nn.Module):\n    def __init__(self, channels):       
super(Block, self).__init__()        
self.op0 = op_id_list[0]       
self.op1 = op_id_list[1]        
self.op2 = op_id_list[2]        
self.op3 = op_id_list[3]        
self.op4 = op_id_list[4]    
self.op5 = op_id_list[5]   
self.op6 = op_id_list[6]    
self.op7 = op_id_list[7]    
self.op8 = op_id_list[8]        
self.op9 = op_id_list[9]    
def forward(self, s0):
s1 = self.op0(s0)
s2 = self.op1(s0) + self.op2(s1)
s3 = self.op3(s0) + self.op4(s1) + self.op5(s2)
s4 = self.op6(s0) + self.op7(s1) + self.op8(s2) + self.op9(s3)    
return s4
Note that each stage can only have gradient flow with up to two previous stages. In other words, s3 can only have gradient flow with two stages from s0, s1, and s2, while the other stage must be 'none'. 
Similarly, s4 can only have gradient flow with two stages from s0, s1, s2, and s3, and the other two stages must be 'none'.
For example, (Model: |skip_connect~0|+|gated_tcn~0|trans~1|+|diff_gcn~0|none~1|skip_connect~2|+|none~0|s_trans~1|none~2|diff_gcn~3|)
means that there are four steps in a model, the first |skip_connect~0| represents the 1st module connecting the 0th module, and the operator used for the connection is 'skip_connect' operator, 
the second |gated_tcn~0|trans~1| represents the 2nd module connecting the 0th module and the 1st module, the operator used  are the 'gated_tcn' operator and the 'trans' operator respectively,
 |diff_gcn~0|none~1|skip_connect~2| represents the 3th module connecting the 0th module and the 2nd module, the operator used  are the 'diff_gcn' operator and the 'skip_connect' operator respectively, 
the operation on the 1st module is 'none' so the 3th module is not connected to the 1st module
|none~0|s_trans~1|none~2|diff_gcn~3| represents the 4th module connecting the 1st module and the 3th module, the operator used  are the 's_trans' operator and the 'diff_gcn' operator respectively, the "
operation on the 0th module and the 2nd module is 'none' so the 4th module is not connected to the 0th module and the 2nd module.   
Here are {length} examples of the top sets of operations gathered from similar benchmark datasets:\n'''
        for i in range(length):
            input+=f"Top-performing model designs from {list(next(iter(similarity_scores.values())).keys())[i]} (Similarity score: {list(next(iter(similarity_scores.values())).values())[i]}):\n"
            input+="".join(lines[i])
        input+='''Please note that the architecture rankings from similar datasets are incomplete. The complete list of architectures is as follows:
1.[[0,x], [0,x], [1,x], [0,x], [1,x], [0,x], [1,x]]
2.[[0,x], [0,x], [1,x], [0,x], [1,x], [0,x], [2,x]]
3.[[0,x], [0,x], [1,x], [0,x], [1,x], [0,x], [3,x]]
4.[[0,x], [0,x], [1,x], [0,x], [1,x], [1,x], [2,x]]
5.[[0,x], [0,x], [1,x], [0,x], [1,x], [1,x], [3,x]]
6.[[0,x], [0,x], [1,x], [0,x], [1,x], [2,x], [3,x]]
7.[[0,x], [0,x], [1,x], [0,x], [2,x], [0,x], [1,x]]
8.[[0,x], [0,x], [1,x], [0,x], [2,x], [0,x], [2,x]]
9.[[0,x], [0,x], [1,x], [0,x], [2,x], [0,x], [3,x]]
10.[[0,x], [0,x], [1,x], [0,x], [2,x], [1,x], [2,x]]
11.[[0,x], [0,x], [1,x], [0,x], [2,x], [1,x], [3,x]]
12.[[0,x], [0,x], [1,x], [0,x], [2,x], [2,x], [3,x]]
13.[[0,x], [0,x], [1,x], [1,x], [2,x], [0,x], [1,x]]
14.[[0,x], [0,x], [1,x], [1,x], [2,x], [0,x], [2,x]]
15.[[0,x], [0,x], [1,x], [1,x], [2,x], [0,x], [3,x]]
16.[[0,x], [0,x], [1,x], [1,x], [2,x], [1,x], [2,x]]
17.[[0,x], [0,x], [1,x], [1,x], [2,x], [1,x], [3,x]]
18.[[0,x], [0,x], [1,x], [1,x], [2,x], [2,x], [3,x]]

You will need to select 8 macro topological architectures from the complete list for the unseen dataset based on the incomplete statistical ranking of the macro topological architectures from similar datasets, so that the MAE value of the downstream model in the spatio-temporal domain is as low as possible.
Please carefully consider the architecture design rules, and make selections with reference to the architecture rankings from similar datasets.
Your answer should be in the following format:\nAnalysis:[Reason]\nFor the unseen dataset: [1.architecture1, 2.architecture2, 3.architecture3, 4.architecture4, 5.architecture5, 6.architecture6, 7.architecture7, 8.architecture8]\n'''
        print("input:",input)
        messages = [
            SystemMessage(content=RoleSetting),
            HumanMessage(content=input)
        ]
        while True:
            response = langchain_query.invoke(messages)   
            print("response.content:", response.content)
            match = re.search(r"For\s+the\s+unseen\s+dataset:\s*(.+)", response.content, re.DOTALL)
            extracted_text = match.group(1)
            print("extracted_text:",extracted_text)
            archs = re.findall(r"\d+\.\s+\*\*(\[\[.*?\]\])\*\*", extracted_text, re.DOTALL)
            if archs:
                print("archs:",archs)
                break
        memory = ConversationBufferMemory(return_messages=True)
        input=RoleSetting+input
        memory.chat_memory.add_user_message(input)
        memory.chat_memory.add_ai_message(response.content)
        return archs,memory

    def Suggestions(selected_op,memory):
        langchain_query = ChatZhipuAI(temperature=0.3, model='glm-4-plus')
        conversation = ConversationChain(llm=langchain_query, memory=memory, verbose=True)
        followup = f"You peers need to rank the five operations except 'none' for the unseen dataset. This is the result obtained by peers regarding the selection of topological connection rules before:{selected_op}. You need to provide some suggestions for the ranking of the operators based on the sorting results.\n Your answer should be in the following format:\n Analysis:[Reason]\n Suggestion: [Suggestion]\n"
        while True:
            op_suggestion = conversation.run(followup)
            print("op_suggestion:", op_suggestion)
            match = re.search(r"Suggestion:(.*)", op_suggestion, re.DOTALL)
            if match:
                suggestion_text = match.group(1).strip()
                op_suggestion = re.sub(r"[#*]", "", suggestion_text)
                print("op_suggestion:",op_suggestion)
                break
        return op_suggestion
    
        #result = extracted_text.group(1).strip() if extracted_text else None
        memory = ConversationBufferMemory(return_messages=True)
        input=RoleSetting+input
        memory.chat_memory.add_user_message(input)
        memory.chat_memory.add_ai_message(response.content)
        conversation = ConversationChain(llm=langchain_query, memory=memory, verbose=True)
        followup = "You need to provide some suggestions for the selection and design of the subsequent topological structure based on the sorting results.\nNote that each stage can only have gradient flow with up to two previous stages. In other words, s3 can only have gradient flow with two stages from s0, s1, and s2, while the other stage must be 'none'. \nSimilarly, s4 can only have gradient flow with two stages from s0, s1, s2, and s3, and the other two stages must be 'none'. Your answer should be in the following format:\n Analysis:[Reason]\n Suggestion: [Suggestion]\n"
        while True:
            op_suggestion = conversation.run(followup)
            print("op_suggestion:", op_suggestion)
            match = re.search(r"Suggestion:(.*)", op_suggestion, re.DOTALL)
            if match:
                suggestion_text = match.group(1).strip()
                op_suggestion = re.sub(r"[#*]", "", suggestion_text)
                print("op_suggestion:",op_suggestion)
                break
        memory.chat_memory.add_user_message(followup)
        memory.chat_memory.add_ai_message(op_suggestion)
        return op_suggestion,memory
        #followup = f"This is the result of the usability ranking of the operations you selected earlier:{selected_operator_string}. You can reanalyze and modify your sorting result based on this sorting result. (If you think the original sorting result is good enough, you don't have to make any modifications.)   Your answer should be in the following format:\n Analysis:[Reason]\nFor the unseen dataset: [1.architecture1, 2.architecture2, 3.architecture3, 4.architecture4, 5.architecture5, 6.architecture6, 7.architecture7, 8.architecture8]\n"
        # while True:
        #     response = conversation.run(followup)
        #     print("response:", response)
        #     match = re.search(r"For\s+the\s+unseen\s+dataset:\s*(.+)", response, re.DOTALL)
        #     extracted_text = match.group(1)
        #     print("extracted_text:",extracted_text)
        #     archs = re.findall(r"\d+\.\s+\*\*(\[\[.*?\]\])\*\*", extracted_text, re.DOTALL)
        #     if archs:
        #         print("archs:",archs)
        #         break
        # followup = "Based on the analysis provided earlier, here are some suggestions for the design of subsequent models, particularly in terms of architecture selection. Your answer should be in the following format:\n Analysis:[Reason]\n Suggestion: [Suggestion]\n"


    def ModifiedConnectionRule(architecture_suggestion,memory):
        langchain_query = ChatZhipuAI(temperature=0.3, model='glm-4-plus')
        conversation = ConversationChain(llm=langchain_query, memory=memory, verbose=True)
        followup = f'''The opinion of your peers on your choice of operator is:{architecture_suggestion}\nYou need to improve your original choice of topological connection. If the opinion remains the same as before, there is no need to modify it. 
Your answer should be in the following format:\nAnalysis:[Reason]\nFor the unseen dataset: [1.architecture1, 2.architecture2, 3.architecture3, 4.architecture4, 5.architecture5, 6.architecture6, 7.architecture7, 8.architecture8]'''
        while True:
            architecture_rank = conversation.run(followup)
            print("architecture_rank11111:", architecture_rank)
            match = re.search(r"For\s+the\s+unseen\s+dataset:\s*(.+)", architecture_rank, re.DOTALL)
            extracted_text = match.group(1)
            print("extracted_text:",extracted_text)
            archs = re.findall(r"\d+\.\s+\*\*(\[\[.*?\]\])\*\*", extracted_text, re.DOTALL)
            if archs:
                print("archs:",archs)
                break
        return archs

