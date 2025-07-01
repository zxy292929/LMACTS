from langchain_community.chat_models import ChatZhipuAI
import requests
import chromadb
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from chromadb.utils import embedding_functions
import re
from openai import OpenAI
import os

class Evaluator:
    def __init__(self):
        self.client = chromadb.Client()
        self.embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")     

    def ReviewModel(self,refined_child,question):
        suggestion=""
        vectorstore = FAISS.load_local(
            "/root/DesiGNN-copy7/rag_database",
             embeddings=self.embedding_fn,
             allow_dangerous_deserialization=True
        )
        architecture_information = next(iter(refined_child.values()))
        print("architecture_information:",architecture_information)
        langchain_query = ChatZhipuAI(temperature=1, model='glm-4-plus')
        prompt=f'''You are a model review expert in the field of Graph Neural Networks(GNN). You need to evaluate a GNN model that answers a Neural Architecture Search(NAS) question. 
For this model, based on the advice of the operation expert and architecture expert selected for this dataset and your experience, give your suggestions for improvements to this model and provide a confidence score for your suggestions.
The NAS question is:
{question}
A response to this question is:{architecture_information}.'''
        results = vectorstore.similarity_search(question, k=5)
        for i, doc in enumerate(results, 1):
             suggestion+=f"\n{i}: {doc.page_content}"
        prompt+=f'''The following are the relevant knowledge fragments retrieved from the papers. Please read them carefully and generate revision suggestions for the current model regarding operations and macro architecture selection in combination with these fragments:{suggestion}
Please review this response and rate your confidence score for your evaluation. The score should be an integer from 1 to 10, with a higher score indicating greater confidence in your assessment.
Your answer should be in the following format:
My review:[Review]
My confidence score for this evaluation:[Score]'''
        print("prompt:",prompt)
        while True:
            suggestion_text = langchain_query.invoke(prompt)
            match = re.search(r"My Review:\s*(.*?)\s*My Confidence Score for This Evaluation", suggestion_text.content, re.DOTALL| re.IGNORECASE) 
            match2 = re.search(r"My Confidence Score for This Evaluation:\s*(\d+)", suggestion_text.content,re.IGNORECASE)
            if match and match2:
                review_text = match.group(1) 
                review = re.sub(r"[*#]", "", review_text)
                confidence_score = int(match2.group(1))
                break

        seggestion=f'''You are an expert in neural architecture search (NAS) for Graph Neural Networks (GNN).You designed an model for a question in the field of Graph Neural Networks and received peer review feedbacks on it. Review consists of review opinions and a confidence score, which is an integer from 1 to 10, with higher confidence scores indicating more confidence in their review. Now, you need to modify the model based on the review comments.
The question is:
{question}
Your response to this question is:{refined_child}.
The review expert's feedback is:
{review}
And his confidence score in this feedback is:{confidence_score}.
You need to improve your model based on the review. (You don't have to change your answers if you think your model is good enough).Remember that there are only four operations in a model
Your answer should be in the following format:
My new model: (Architecture: [TBD], Operations: [TBD])
Reasons:
'''     
        print("seggestion:",seggestion)
        while True:
            langchain_query = ChatZhipuAI(temperature=1, model='glm-4-plus')
            modified_text = langchain_query.invoke(seggestion)
            match = re.search(r"My new model:\s*(.*?)\s*Reasons:", modified_text.content, re.DOTALL| re.IGNORECASE)
            if match:
                modified_model = match.group(1) 
                arch_match = re.search(r'Architecture: (\[.*?\])', modified_model)
                op_match = re.search(r'Operations: (\[.*?\])', modified_model)
                if arch_match and op_match:
                    arch_list = eval(arch_match.group(1))  
                    op_list = eval(op_match.group(1)) 
                    ops_length = len(op_list)
                    if ops_length == 4:
                        break
        modified_model = {'link': arch_list, 'ops': op_list}
        return modified_model
    


    @staticmethod
    def extract_model_designs(llm_response, dataset_name):
        """
        Extracts model designs suggested by the LLM for each source dataset.

        :param llm_response: A string containing the LLM's response in specified formats.
        :param dataset_name: The name of the source dataset used in the query.
        :return: A dictionary with source dataset names as keys and their suggested model designs as values.
        """
        # Patterns to match both formats of the LLM's response
        patterns = [
            r"\(Architecture: (\[.*?\]), Operations: (\[.*?\])\)",
            r"-\s*\*\*Architecture:\*\*\s*(\[.*?\])\s*-\s*\*\*Operations:\*\*\s*(\[.*?\])",
            r"-\s*\*\*Architecture\*\*:\s*(\[.*?\])\s*-\s*\*\*Operations\*\*:\s*(\[.*?\])",
            r"-\s*\*\*Architecture:\s*(\[.*?\])\*\*\s*-\s*\*\*Operations:\s*(\[.*?\])\*\*"
        ]

        # Initialize a dictionary to hold the extracted designs
        suggested_designs = {}

        for pattern in patterns:
            # Find all matches in the response
            matches = re.findall(pattern, llm_response)
            # Iterate through all matches and populate the dictionary
            for match in matches:
                architecture, operations = match
                # Evaluate to convert string representations to actual lists
                architecture_eval = eval(architecture)
                operations_eval = eval(operations)
                # Populate the dictionary with the specific source dataset and extracted values
                suggested_designs = {
                    "Architecture": architecture_eval,
                    "Operations": operations_eval
                }
                return suggested_designs

        return suggested_designs