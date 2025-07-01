# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license

from ast import Not
import re
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field, create_model, conlist
import ast

class LLMPromptConfigurator:

    @staticmethod
    def generate_design_suggestion_prompt_parser(dataset_name, models_info, similarities=None):
        """
        Generates a prompt asking the LLM for a model design suggestion.
        :param dataset_name: The source dataset. 
        :param models_info: The top model designs.
        :param similarities: Optional dictionary of similarity scores.
        :return: A formatted LLM prompt string.
        """
        prompt = ChatPromptTemplate.from_messages(
            [("system", "You are a machine learning expert proficient in Graph Neural Networks (GNN) design and graph "
                        "dataset understanding. Your task is to recommend a GNN model architecture that performs well "
                        "on the unseen dataset based on the top-performing and bad-performing GNN model architectures on the most similar "
                        "benchmark dataset to the user.\n"
                        "In the context of GNN, the design of a model is described by two main components: the "
                        "operation list and the macro architecture list. Here are the detailed settings:\n"
                        "1. The operation list is a list of four strings. We consider 9 candidate operations, which "
                        "are:\n"
                        "- 'gat': Graph Attention Network layer, utilizing attention mechanisms to weigh the "
                        "importance of nodes' neighbors.\n"
                        "- 'gcn': Graph Convolutional Network layer, applying a convolutional operation over the "
                        "graph to aggregate neighborhood information.\n"
                        "- 'gin': Graph Isomorphism Network layer, designed to capture the graph structure in the "
                        "embedding.\n"
                        "- 'cheb': Chebyshev Spectral Graph Convolution, using Chebyshev polynomials to filter graph "
                        "signals.\n"
                        "- 'sage': GraphSAGE, sampling and aggregating features from a node's neighborhood.\n"
                        "- 'arma': ARMA layer, utilizing Auto-Regressive Moving Average filters for graph "
                        "convolution.\n"
                        "- 'graph': k-GNN, extending the GNN to capture k-order graph motifs.\n"
                        "- 'fc': Fully Connected layer, a dense layer that does not utilize graph structure.\n"
                        "- 'skip': Skip Connection, enabling the creation of residual connections.\n"
                        "For example, an operation list could be ['gcn', 'gin', 'fc', 'cheb'], with 'gcn' as the first "
                        "computing node. The order of operations in the list matters. \n"
                        "2. The macro architecture list is represented as a directed acyclic graph (DAG), dictating "
                        "the flow of data through various operations. Since we constrain the DAG of the computation "
                        "graph to have only one input node for each intermediate node, the macro space can be "
                        "described by a list of four integers. The integer of each position represents the input "
                        "source of the operation at the corresponding position in the operation list. For example, "
                        "the integer 0 at position 1 means the corresponding operation at position 1 of the operation "
                        "list uses raw input as input, while the integer 1 at position 3 means the corresponding "
                        "operation at position 3 of the operation list uses the first computing node (the operation "
                        "at position 0 of the operation list) as input. We consider 9 distinct DAG configurations in "
                        "our search space, which are:\n"
                        "- [0, 0, 0, 0]: All operations in the operation list take the raw input directly, creating "
                        "parallel pathways right from the start, allowing for multiple independent transformations of "
                        "the input data.\n"
                        "- [0, 0, 1, 1]: The first two operations in the operation list process the raw input in "
                        "parallel. The third and fourth operations are parallel, both applying transformations to the "
                        "output of the first operation.\n"
                        "- [0, 0, 1, 2]: The first two operations in the operation list are parallel, and the third "
                        "operation processes the output of the first operation. The fourth operation then applies a "
                        "transformation to the output of the second operation, creating a mix of parallel and "
                        "sequential flows.\n"
                        "- [0, 0, 1, 3]: The first two operations in the operation list process the raw input in "
                        "parallel. The third operation processes the output of the first operation. The fourth "
                        "operation extends the sequence by processing the output of the third operation, showcasing a "
                        "blend of parallel processing at the start followed by a sequential chain.\n"
                        "- [0, 1, 1, 1]: The first operation in the operation list processes the raw input, while the "
                        "next three operations process the output of the first operation in parallel, allowing for "
                        "diverse transformations of the same set of features.\n"
                        "- [0, 1, 1, 2]: The first operation in the operation list processes the raw input, while the "
                        "next two operations process the output of the first operation in parallel. The fourth "
                        "operation then processes the output of the second operation, introducing a sequential "
                        "element within a primarily parallel structure.\n"
                        "- [0, 1, 2, 2]: The first operation in the operation list processes the raw input, the "
                        "second operation processes the output of the first operation, and the third and fourth "
                        "operations both apply transformations to the output of the second operation in parallel, "
                        "creating a divergent path after a single sequence.\n"
                        "- [0, 1, 2, 3]: Represents a fully sequential architecture where each operation receives the "
                        "output of the previous operation, forming a linear sequence of transformations from the raw "
                        "input to the final output.\n"
                        "Together, the operation list and the macro architecture list define the computation graph of "
                        "a GNN, including the flow of data through various operations. For example, the model design "
                        "(Architecture: [0, 1, 1, 3], Operations: ['gcn', 'cheb', 'gin', 'fc']) represents a GNN "
                        "architecture where the raw input first undergoes a GCN operation. Subsequently, the output "
                        "of the GCN is processed by the second Chebyshev convolution and the third GIN operations in "
                        "parallel pathways. The fourth operation, the Fully Connected layer, processes the output of "
                        "the GIN operation. The outputs of the second Chebyshev convolution and the Fully Connected "
                        "layer are concatenated together before producing the final output. When seeing a GNN model "
                        "design of this format, you need to understand the actual operations they represent and how "
                        "they are connected."),
             ("user", "{input}")]
        )
        bad_model = any(model.get('bad_models') for model in models_info)
        if bad_model:
            user_input = "Based on the given most similar benchmark dataset and its corresponding top-performing and " \
                         "bad-performing GNN model architectures below, please take a deep breath and work on this " \
                         "problem step-by-step: analyze the potential patterns or underlying principles in the " \
                         "operation lists and the macro architecture lists of the top-performing and bad-performing " \
                         "model designs. This may include commonalities in the choice of operations, preferences for " \
                         "certain macro architecture configurations, or any recurring themes that might indicate a " \
                         "successful or failure approach to constructing GNN architectures for similar types of " \
                         "data. After evaluating these patterns, you need to use your comprehensive knowledge to " \
                         "suggest an optimal model design for the unseen dataset. You should think about how " \
                         "specific operations and macro architecture designs have contributed to high performance " \
                         "in similar datasets. Your suggestion should reflect a thoughtful synthesis of these " \
                         "insights, aiming to capture the most effective elements in the provided top-performing " \
                         "designs and avoid the most ineffective elements in the bad-performing designs. Here are " \
                         "the top-performing and bad-performing designs:\n"
        else:
            user_input = "Based on the given most similar benchmark dataset and its corresponding top-performing GNN " \
                         "model architectures below, please take a deep breath and work on this problem step-by-step: " \
                         "analyze the potential patterns or underlying principles in the operation lists and the macro " \
                         "architecture lists of the top-performing model designs. This may include commonalities in the " \
                         "choice of operations, preferences for certain macro architecture configurations, or any " \
                         "recurring themes that might indicate a successful approach to constructing GNN architectures " \
                         "for similar types of data. After evaluating these patterns, you need to use your comprehensive " \
                         "knowledge to suggest an optimal model design for the unseen dataset. You should think about " \
                         "how specific operations and macro architecture designs have contributed to high performance in " \
                         "similar datasets. Your suggestion should reflect a thoughtful synthesis of these insights, " \
                         "aiming to capture the most effective elements in the provided top-performing designs. Here are " \
                         "the top-performing designs:\n"

        for model_info in models_info:
            selected_dataset = model_info['selected_dataset']
            if similarities and selected_dataset in similarities[dataset_name]:
                user_input += f"Top-performing model designs from {selected_dataset} (Similarity score: {similarities[dataset_name][selected_dataset]}):\n"
            else:
                user_input += f"Top-performing model designs from {selected_dataset}:\n"
            for model_design in model_info['top_models']:
                link_structure, operations = model_design
                user_input += f"- (Architecture: {link_structure}, Operations: {operations})\n"

            if bad_model:
                user_input += f"Bad-performing model designs from {selected_dataset}:\n"
                for model_design in model_info['bad_models']:
                    link_structure, operations = model_design
                    user_input += f"- (Architecture: {link_structure}, Operations: {operations})\n"

        fields = {}
        fields[f"initial_operation"] = (Optional[conlist(str, min_items=4, max_items=4)],
                                                         Field(default=None,
                                                               description=f"The operation list of the optimal model "
                                                                           f"design suggested for the unseen dataset "
                                                                           f"{dataset_name}."))
        fields[f"initial_macro"] = (Optional[conlist(int, min_items=4, max_items=4)],
                                                     Field(default=None,
                                                           description=f"The macro architecture list of the optimal "
                                                                       f"model design suggested for the unseen dataset "
                                                                       f"{dataset_name}."))
        fields[f"initial_design_reason"] = (Optional[str],
                                                             Field(default=None,
                                                                   description=f"Reason for the optimal model design "
                                                                               f"suggested for the unseen dataset "
                                                                               f"{dataset_name}."))
        initialization_tool = create_model('InitialModelDesign', **fields)
        if bad_model:
            initialization_tool.__doc__ = "Suggest an optimal GNN model architecture on the unseen dataset based on the " \
                                          "top-performing and bad-performing GNN model architectures on the most " \
                                          "similar benchmark dataset."
        else:
            initialization_tool.__doc__ = "Suggest an optimal GNN model architecture on the unseen dataset based on the " \
                                          "top-performing GNN model architectures on the most similar benchmark dataset."

        print(prompt)
        print(user_input)

        return prompt, user_input, initialization_tool

    @staticmethod
    def generate_simple_design_suggestion_prompt_parser(dataset_name, description=None):
        """
        Generates a prompt asking the LLM for a model design suggestion.
        :param dataset_name: The source dataset.
        :return: A formatted LLM prompt string.
        """
        prompt = ChatPromptTemplate.from_messages(
            [("system", "You are a machine learning expert proficient in Graph Neural Networks (GNN) design and graph "
                        "dataset understanding. Your task is to recommend a GNN model architecture that performs well "
                        "on the unseen dataset to the user based on the dataset description.\n"
                        "In the context of GNN, the design of a model is described by two main components: the "
                        "operation list and the macro architecture list. Here are the detailed settings:\n"
                        "1. The operation list is a list of four strings. We consider 9 candidate operations, which "
                        "are:\n"
                        "- 'gat': Graph Attention Network layer, utilizing attention mechanisms to weigh the "
                        "importance of nodes' neighbors.\n"
                        "- 'gcn': Graph Convolutional Network layer, applying a convolutional operation over the "
                        "graph to aggregate neighborhood information.\n"
                        "- 'gin': Graph Isomorphism Network layer, designed to capture the graph structure in the "
                        "embedding.\n"
                        "- 'cheb': Chebyshev Spectral Graph Convolution, using Chebyshev polynomials to filter graph "
                        "signals.\n"
                        "- 'sage': GraphSAGE, sampling and aggregating features from a node's neighborhood.\n"
                        "- 'arma': ARMA layer, utilizing Auto-Regressive Moving Average filters for graph "
                        "convolution.\n"
                        "- 'graph': k-GNN, extending the GNN to capture k-order graph motifs.\n"
                        "- 'fc': Fully Connected layer, a dense layer that does not utilize graph structure.\n"
                        "- 'skip': Skip Connection, enabling the creation of residual connections.\n"
                        "For example, an operation list could be ['gcn', 'gin', 'fc', 'cheb'], with 'gcn' as the first "
                        "computing node. The order of operations in the list matters. \n"
                        "2. The macro architecture list is represented as a directed acyclic graph (DAG), dictating "
                        "the flow of data through various operations. Since we constrain the DAG of the computation "
                        "graph to have only one input node for each intermediate node, the macro space can be "
                        "described by a list of four integers. The integer of each position represents the input "
                        "source of the operation at the corresponding position in the operation list. For example, "
                        "the integer 0 at position 1 means the corresponding operation at position 1 of the operation "
                        "list uses raw input as input, while the integer 1 at position 3 means the corresponding "
                        "operation at position 3 of the operation list uses the first computing node (the operation "
                        "at position 0 of the operation list) as input. We consider 9 distinct DAG configurations in "
                        "our search space, which are:\n"
                        "- [0, 0, 0, 0]: All operations in the operation list take the raw input directly, creating "
                        "parallel pathways right from the start, allowing for multiple independent transformations of "
                        "the input data.\n"
                        "- [0, 0, 1, 1]: The first two operations in the operation list process the raw input in "
                        "parallel. The third and fourth operations are parallel, both applying transformations to the "
                        "output of the first operation.\n"
                        "- [0, 0, 1, 2]: The first two operations in the operation list are parallel, and the third "
                        "operation processes the output of the first operation. The fourth operation then applies a "
                        "transformation to the output of the second operation, creating a mix of parallel and "
                        "sequential flows.\n"
                        "- [0, 0, 1, 3]: The first two operations in the operation list process the raw input in "
                        "parallel. The third operation processes the output of the first operation. The fourth "
                        "operation extends the sequence by processing the output of the third operation, showcasing a "
                        "blend of parallel processing at the start followed by a sequential chain.\n"
                        "- [0, 1, 1, 1]: The first operation in the operation list processes the raw input, while the "
                        "next three operations process the output of the first operation in parallel, allowing for "
                        "diverse transformations of the same set of features.\n"
                        "- [0, 1, 1, 2]: The first operation in the operation list processes the raw input, while the "
                        "next two operations process the output of the first operation in parallel. The fourth "
                        "operation then processes the output of the second operation, introducing a sequential "
                        "element within a primarily parallel structure.\n"
                        "- [0, 1, 2, 2]: The first operation in the operation list processes the raw input, the "
                        "second operation processes the output of the first operation, and the third and fourth "
                        "operations both apply transformations to the output of the second operation in parallel, "
                        "creating a divergent path after a single sequence.\n"
                        "- [0, 1, 2, 3]: Represents a fully sequential architecture where each operation receives the "
                        "output of the previous operation, forming a linear sequence of transformations from the raw "
                        "input to the final output.\n"
                        "Together, the operation list and the macro architecture list define the computation graph of "
                        "a GNN, including the flow of data through various operations. For example, the model design "
                        "(Architecture: [0, 1, 1, 3], Operations: ['gcn', 'cheb', 'gin', 'fc']) represents a GNN "
                        "architecture where the raw input first undergoes a GCN operation. Subsequently, the output "
                        "of the GCN is processed by the second Chebyshev convolution and the third GIN operations in "
                        "parallel pathways. The fourth operation, the Fully Connected layer, processes the output of "
                        "the GIN operation. The outputs of the second Chebyshev convolution and the Fully Connected "
                        "layer are concatenated together before producing the final output. When seeing a GNN model "
                        "design of this format, you need to understand the actual operations they represent and how "
                        "they are connected."),
             ("user", "{input}")]
        )
        user_input = "Based on the following dataset description, please take a deep breath and work on this problem " \
                     "step-by-step: use your comprehensive knowledge to suggest an optimal model design for the " \
                     "unseen dataset. You should think about how specific operations and macro architecture designs " \
                     "could potentially contribute to high performance on the unseen dataset. Here is the dataset " \
                     "description:\n"
        user_input += description

        fields = {}
        fields[f"initial_operation"] = (Optional[conlist(str, min_items=4, max_items=4)],
                                                         Field(default=None,
                                                               description=f"The operation list of the optimal model "
                                                                           f"design suggested for the unseen dataset."))
        fields[f"initial_macro"] = (Optional[conlist(int, min_items=4, max_items=4)],
                                                     Field(default=None,
                                                           description=f"The macro architecture list of the optimal "
                                                                       f"model design suggested for the unseen dataset."))
        fields[f"initial_design_reason"] = (Optional[str],
                                                             Field(default=None,
                                                                   description=f"Reason for the optimal model design "
                                                                               f"suggested for the unseen dataset."))
        initialization_tool = create_model('InitialModelDesign', **fields)
        initialization_tool.__doc__ = "Suggest an optimal GNN model architecture on the unseen dataset based on the " \
                                      "dataset description."

        return prompt, user_input, initialization_tool

    @staticmethod
    def generate_design_suggestion_prompt(dataset_name, models_info, selected_operator,architecture_rank,ar_suggestion,model_suggestion,similarities=None, description=None):
        """
        Generates a prompt asking the LLM for a model design suggestion.
        :param dataset_name: The source dataset.
        :param models_info: The top model designs.
        :param similarities: Optional dictionary of similarity scores.
        :return: A formatted LLM prompt string.
        """
        prompt=""
        selected_operator = ' '.join([f"{i+1}.{op}" for i, op in enumerate(selected_operator)])
        architecture_rank_string=' '.join([f"{i+1}.{op}" for i, op in enumerate(architecture_rank)])
        architecture_suggestion = ""
        for arch_str in architecture_rank:
            arch_str_clean = arch_str.replace('x', '0')
            link_structure = ast.literal_eval(arch_str_clean)

            s1 = [link_structure[0]]                     # s1 = op0(s0)
            s2 = link_structure[1:3]                     # s2 = op1(s0) + op2(s1)
            s3 = link_structure[3:5]                     # s3 = op3(s0) + op4(s2)
            s4 = link_structure[5:]                      # s4 = op5(s0) + op6(s1) + op7(s2) + op8(s3)

            def stage_to_str(inputs, stage_id):
                node_inputs = {i: f"none~{i}" for i in range(stage_id)}  # 默认都为 none
                for from_idx, _ in inputs:
                    node_inputs[from_idx] = f"operation~{from_idx}"
                joined = '|'.join(node_inputs[i] for i in range(stage_id))
                return f"|{joined}|"

            s1_str = stage_to_str(s1, 1)
            s2_str = stage_to_str(s2, 2)
            s3_str = stage_to_str(s3, 3)
            s4_str = stage_to_str(s4, 4)

            final_result = f"- (Architecture: {s1_str}+{s2_str}+{s3_str}+{s4_str})\n"
            architecture_suggestion += final_result

        # Introduction to model design components
        if models_info:
            bad_model = any(model.get('bad_models') for model in models_info)
            if bad_model:
                intro = ("You are Quoc V. Le, a computer scientist and artificial intelligence researcher who is widely regarded as one of the leading experts in deep learning and neural network architecture search.  Your work in"
                     "this area has focused on developing efficient algorithms for searching the space of possible neural network architectures, with the goal of finding architectures that perform well on a given task "
                     "while minimizing the computational cost of training and inference."
                     "The task at hand involves leveraging the best model design knowledge and practices from similar benchmark datasets in the field of spatiotemporal neural networks. By examining top-performing and bad-performing models on these datasets, we aim to quickly recommend an optimal model design for an unseen dataset, ensuring good performance with minimal initial experimentation, the goal is for the model to get the lowest possible MAE value when tested. ")
            else:
                intro = ("You are Quoc V. Le, a computer scientist and artificial intelligence researcher who is widely regarded as one of the leading experts in deep learning and neural network architecture search.  Your work in"
                     "this area has focused on developing efficient algorithms for searching the space of possible neural network architectures, with the goal of finding architectures that perform well on a given task "
                     "while minimizing the computational cost of training and inference."
                     "The task at hand involves leveraging the best model design knowledge and practices from similar benchmark datasets in the field of spatiotemporal neural networks. By examining top-performing models on these datasets, we aim to quickly recommend an optimal model design for an unseen dataset, ensuring good performance with minimal initial experimentation, the goal is for the model to get the lowest possible MAE value when tested. ")
        else:
            intro = ("You are Quoc V. Le, a computer scientist and artificial intelligence researcher who is widely regarded as one of the leading experts in deep learning and neural network architecture search.  Your work in"
                     "this area has focused on developing efficient algorithms for searching the space of possible neural network architectures, with the goal of finding architectures that perform well on a given task "
                     "while minimizing the computational cost of training and inference."
                     " The task at hand involves leveraging the best model design knowledge and practices from your knowledge in the field of Spatiotemporal Neural Networks.  By examining top-performing models on these "
                     "datasets, we aim to quickly recommend an optimal model design for an unseen dataset and maximize the model's performance, ensuring good performance with minimal initial experimentation.  (The metric "
                     "of the model's performance is MAE value ). ")
        
        intro += ("\nIn the context of spatiotemporal neural networks, the design of a model is described by two main components: the operation list and the macro topological architecture. Here are the detailed settings:\n"
                  "The 6 available operations are as follows:\n"
                  "- 'skip_connect': Skip Connection, a mechanism to bypass intermediate layers, preserving original input information and supporting residual learning. \n"
                  "- 'gated_tcn': The Gated TCN (Gated Temporal Convolutional Network) Layer, consists of two parallel temporal convolution layers, TCN-a and TCN-b. These layers process the input data using the tanh and sigmoid activation functions, respectively, to extract temporal features. The processed outputs are fused through a multiplication operation, and this gating mechanism allows the model to dynamically adjust the flow of information, enhancing its ability to capture important temporal features. Additionally, the output of the Gated TCN module is added to the original input via a skip connection, forming a residual connection. This helps alleviate the vanishing gradient problem in deep network training and enables the network to more effectively learn long-term dependencies. \n"
                  "- 'diff_gcn': Diffusion Graph Convolutional Network Layer, extends graph convolution by modeling the diffusion process of node signals over multiple steps.  It aggregates information from a node's neighbors and higher-order neighbors, capturing long-range dependencies and temporal dynamics for tasks like spatial-temporal forecasting. \n"
                  "- 'trans': Transformer, uses self-attention to model relationships between stages, effective for capturing global interactions.\n"
                  "- 's_trans': Spatial Transformer Layer, combines spatial attention and convolutional operations to process input data with a spatial-temporal structure.  It first applies a spatial attention mechanism to capture long-range dependencies between nodes, using a modified attention layer (SpatialAttentionLayer).  Then, the layer performs two 1D convolution operations to transform the feature dimensions.  Layer normalization and dropout are applied to stabilize training and prevent overfitting.  Finally, the output is reshaped and permuted to match the required format for further processing.  This layer is designed to handle high-dimensional time-series or graph-structured data, enabling the model to focus on important spatial features while learning efficient representations. \n"
                  "- 'none': This operation indicates that there is no operational connection between the two stages, it breaks the gradient flow between two stages.\n"
                  "The macro topological architecture design is as follows:\n"
                  "The neural network block is defined by 10 operations (i.e., op_list = [op0, op1, op2, op3, op4, op5, op6, op7, op8, op9]), which represent the operations executed between various stages of the block.  This block comprises 5 "
                  "stages, labeled as s0, s1, s2, s3 and s4, each corresponding to distinct feature maps in the neural network."
                  "s0 serves as the input feature map for this block.\n"
                  "s1 will be calculated by s1 = op0(s0).\n"
                  "s2 will be calculated by s2 = op1(s0) + op2(s1).\n"
                  "s3 will be calculated by s3 = op3(s0) + op4(s1) + op5(s2).\n"
                  "s4 will be calculated by s4 = op6(s0) + op7(s1) + op8(s2) + op9(s3).\n"
                  "Note that s4 becomes the output for this block and serves as the input for the subsequent block.\n\n"
                  "Then the implementation of the block will be:\n"
                  "class Block(nn.Module):\n    def __init__(self, channels):\n"        
                  "super(Block, self).__init__()\n"        
                  "self.op0 = op_id_list[0]\n"        
                  "self.op1 = op_id_list[1]\n"        
                  "self.op2 = op_id_list[2]\n"        
                  "self.op3 = op_id_list[3]\n"        
                  "self.op4 = op_id_list[4]\n"    
                  "self.op5 = op_id_list[5]\n"    
                  "self.op6 = op_id_list[6]\n"    
                  "self.op7 = op_id_list[7]\n"    
                  "self.op8 = op_id_list[8]\n"        
                  "self.op9 = op_id_list[9]\n\n"    
                  "def forward(self, s0):\n"        
                  "s1 = self.op0(s0)\n"        
                  "s2 = self.op1(s0) + self.op2(s1)\n"        
                  "s3 = self.op3(s0) + self.op4(s1) + self.op5(s2)\n"    
                  "s4 = self.op6(s0) + self.op7(s1) + self.op8(s2) + self.op9(s3)\n"            
                  "return s4\n\n"
                  "Note that each stage can only have gradient flow with up to two previous stages. In other words, s3 can only have gradient flow with two stages from s0, s1, and s2, while the other stage must be 'none'. "
                  "Similarly, s4 can only have gradient flow with two stages from s0, s1, s2, and s3, and the other two stages must be 'none'.\n\n"
                  "For example, (Architecture: |skip_connect~0|+|gated_tcn~0|trans~1|+|diff_gcn~0|none~1|skip_connect~2|+|none~0|s_trans~1|none~2|diff_gcn~3|)"
                  "means that there are four steps in a model, the first |skip_connect~0| represents the 1st module connecting the 0th module, and the operator used for the connection is 'skip_connect' operator, "
                  "the second |gated_tcn~0|trans~1| represents the 2nd module connecting the 0th module and the 1st module, the operator used  are the 'gated_tcn' operator and the 'trans' operator respectively,"
                  " |diff_gcn~0|none~1|skip_connect~2| represents the 3th module connecting the 0th module and the 2nd module, the operator used  are the 'diff_gcn' operator and the 'skip_connect' operator respectively, "
                  "the operation on the 1st module is 'none' so the 3th module is not connected to the 1st module"
                  "|none~0|s_trans~1|none~2|diff_gcn~3| represents the 4t h module connecting the 1st module and the 3th module, the operator used  are the 's_trans' operator and the 'diff_gcn' operator respectively, the "
                  "operation on the 0th module and the 2nd module is 'none' so the 4th module is not connected to the 0th module and the 2nd module. "
                  "Let's break this down step by step:\n\n"
                  "First, please analyze the 6 available operations.\n\n"
                  "Next, please consider the gradient flow based on the Block class implementation. For example, how the gradient from the later stage affects the earlier stage.\n\n"
                  "Now, answer the question - how we can design a high-performance block using the available operations?\n\n"
                  f"Note that the following operation ranking is selected by operation experts and reflects the operations (excluding 'none') that most effectively improve model performance. Please refer to this ranking when recommending architectures — for example, higher-ranked operations can be used more frequently: {selected_operator}.\n"
                  f"Note that the following 8 macro architectures are selected by architecture selection experts to improve the performance of the model the most, please try to use one of these 8 macro architectures to recommend the architecture after careful analysis:\n{architecture_suggestion}."
                  "Based the analysis, your task is to propose a block design with the given operations that prioritizes performance, without considering factors such as size and complexity, and drawing from the top model "
                  "designs in the similar dataset as a reference.\n\n")
                  

        if models_info:
            if bad_model:
                sim_dataset_prompt = "You will need to recommend an optimal model design for the unseen dataset based on the following top and bad model designs from similar datasets. Here are the top and bad model designs gathered from similar benchmark datasets:\n"
            else:
                sim_dataset_prompt = "You will need to recommend an optimal model design for the unseen dataset based on the following top model designs from similar datasets. Here are the top model designs gathered from similar benchmark datasets:\n"

            sim_dataset_prompt += f"For the unseen dataset:\n"
            '''
            for model_info in models_info:
                selected_dataset = model_info['selected_dataset']
                if similarities and selected_dataset in similarities[dataset_name]:
                    prompt += f"Similarity score to {selected_dataset}: {similarities[dataset_name][selected_dataset]}\n"
                for model_design in model_info['top_models']:
                    link_structure, operations = model_design
                    prompt += f"- From '{selected_dataset}': (Architecture: {link_structure}, Operations: {operations})\n"

                if len(model_info['bad_models']) > 0:
                    prompt += f"Here are the bad model designs from {selected_dataset} that may not perform well:\n"
                    for model_design in model_info['bad_models']:
                        link_structure, operations = model_design
                        prompt += f"- From '{selected_dataset}': (Architecture: {link_structure}, Operations: {operations})\n"
                prompt += "\n"
            '''
            for model_info in models_info:
                selected_dataset = model_info['selected_dataset']
                if similarities and selected_dataset in similarities[dataset_name]:
                    sim_dataset_prompt += f"Top-performing model designs from {selected_dataset} (Similarity score: {similarities[dataset_name][selected_dataset]}):\n"
                else:
                    sim_dataset_prompt += f"Top-performing model designs from {selected_dataset}:\n"
                
                # List out top-performing model designs
                for model_design in model_info['top_models']:
                    link_structure, operations = model_design

                    part1 = link_structure[:1] 
                    part2 = link_structure[1:3]  
                    part3 = link_structure[3:5]  
                    part4 = link_structure[5:]  

                    for i in range(3):
                        if not any(item[0] == i for item in part3):
                            part3.append([i, 5])  

                    for i in range(4):
                        if not any(item[0] == i for item in part4):
                            part4.append([i, 5])  
                    part3 = sorted(part3, key=lambda x: x[0])
                    part4 = sorted(part4, key=lambda x: x[0])
                    arch = part1 + part2 + part3 + part4
                    mapping = {
                        0: "skip_connect",
                        1: "gated_tcn",
                        2: "diff_gcn",
                        3: "trans",
                        4: "s_trans",
                        5: "none"
                    }
                    result = []
                    group = []  

                    for i in range(len(arch)):
                        part = arch[i]
                        first_element = part[0]  
                        second_element = part[1]  
                        action = mapping[second_element]
                        group.append(f"|{action}~{first_element}|")
                        if i + 1 < len(arch) and arch[i + 1][0] != 0:
                            result.append("+".join(group))
                            group = [] 

                    if group:
                        result.append("+".join(group))

                    final_result = "".join(result)
                    final_result = final_result.replace("||", "|")
                    #$prompt += f"- (Architecture: {link_structure}, Operations: {operations})\n"
                    sim_dataset_prompt += f"- (Architecture: {final_result})\n"

                # List out bad-performing model designs
                if bad_model:
                    sim_dataset_prompt += f"Bad-performing model designs from {selected_dataset}:\n"
                    for model_design in model_info['bad_models']:
                        link_structure, operations = model_design
                        sim_dataset_prompt += f"- (Architecture: {link_structure}, Operations: {operations})\n"
        else:
            prompt = "You will need to recommend an optimal model design for the unseen dataset based on the following description: "
            if description:
                prompt += description + '\n'

        if models_info:
            if bad_model:
                sim_dataset_prompt += ("Based on the insights from similar benchmark datasets, consider the potential patterns or underlying principles in the top and bad model designs. This includes commonalities in the choice of operations, preferences for certain macro architecture configurations, or any recurring themes that might indicate a successful approach to constructing spatiotemporal architectures for similar types of data. Evaluate these patterns and, using your comprehensive analysis, suggest an optimal model design for the source dataset. Consider how specific operations and architecture designs have contributed to high performance in similar datasets. Your suggestion should reflect a thoughtful synthesis of these insights, aiming to capture the most effective elements of the provided designs and avoid the most ineffective elements. Additionally, pay attention to the similarity scores between datasets, if provided, to gauge the relevance of each design's features in relation to the source dataset.\n")
            else:
                sim_dataset_prompt += (
#                     "Example: - Architecture: [[0, 4], [0, 1], [1, 2], [1, 3], [2, 2], [0, 0], [3, 0]], Operations: [[0, 's_trans'], [0, 'dcc_1'], [1, 'diff_gcn'], [1, 'trans'], [2, 's_trans'], [0, 'skip_connect'], [3, 'skip_connect']]"
# " - Architecture: [[0, 4], [0, 3], [1, 4], [0, 3], [2, 3], [1, 2], [3, 3]], Operations: [[0, 's_trans'], [0, 'trans'], [1, 's_trans'], [0, 'trans'], [2, 'trans'], [1, 'diff_gcn'], [3, 'trans']]"
#  "- Architecture: [[0, 1], [0, 1], [1, 3], [0, 4], [2, 3], [1, 1], [3, 3]], Operations: [[0, 'dcc_1'], [0, 'dcc_1'], [1, 'trans'], [0, 's_trans'], [2, 'trans'], [1, 'dcc_1'], [3, 'trans']]"
#  "- Architecture: [[0, 2], [0, 3], [1, 1], [1, 1], [2, 3], [2, 1], [3, 3]], Operations: [[0, 'diff_gcn'], [0, 'trans'], [1, 'dcc_1'], [1, 'dcc_1'], [2, 'trans'], [2, 'dcc_1'], [3, 'trans']]"
#  "- Architecture: [[0, 4], [0, 1], [1, 2], [1, 2], [2, 2], [1, 3], [3, 3]], Operations: [[0, 's_trans'], [0, 'dcc_1'], [1, 'diff_gcn'], [1, 'diff_gcn'], [2, 'diff_gcn'], [1, 'trans'], [3, 'trans']]"
#  "- Architecture: [[0, 4], [0, 2], [1, 2], [0, 1], [2, 0], [1, 3], [3, 0]], Operations: [[0, 's_trans'], [0, 'diff_gcn'], [1, 'diff_gcn'], [0, 'dcc_1'], [2, 'skip_connect'], [1, 'trans'], [3, 'skip_connect']]"
#  "- Architecture: [[0, 1], [0, 3], [1, 0], [1, 3], [2, 0], [1, 4], [3, 3]], Operations: [[0, 'dcc_1'], [0, 'trans'], [1, 'skip_connect'], [1, 'trans'], [2, 'dcc_1'], [1, 's_trans'], [3, 'trans']]"
#  "- Architecture: [[0, 4], [0, 3], [1, 0], [0, 2], [2, 0], [1, 4], [3, 0]], Operations: [[0, 's_trans'], [0, 'trans'], [1, 'skip_connect'], [0, 'diff_gcn'], [2, 'skip_connect'], [1, 's_trans'], [3, 'skip_connect']]"
#  "- Architecture: [[0, 2], [0, 3], [1, 0], [1, 2], [2, 2], [0, 0], [3, 0]], Operations: [[0, 'diff_gcn'], [0, 'trans'], [1, 'skip_connect'], [1, 'diff_gcn'], [2, 'diff_gcn'], [0, 'skip_connect'], [3, 'skip_connect']]"
#  "- Architecture: [[0, 2], [0, 4], [1, 2], [0, 3], [2, 3], [2, 1], [3, 0]], Operations: [[0, 'diff_gcn'], [0, 's_trans'], [1, 'diff_gcn'], [0, 'trans'], [2, 'trans'], [2, 'dcc_1'], [3, 'skip_connect']]"
#  "- Architecture: [[0, 3], [0, 3], [1, 3], [1, 2], [2, 0], [1, 1], [3, 0]], Operations: [[0, 'trans'], [0, 'trans'], [1, 'trans'], [1, 'diff_gcn'], [2, 'skip_connect'], [1, 'dcc_1'], [3, 'skip_connect']]"
#  "- Architecture: [[0, 0], [0, 1], [1, 3], [0, 1], [2, 2], [1, 0], [3, 4]], Operations: [[0, 'skip_connect'], [0, 'dcc_1'], [1, 'trans'], [0, 'dcc_1'], [2, 'diff_gcn'], [1, 'skip_connect'], [3, 's_trans']]"
#  "- Architecture: [[0, 0], [0, 0], [1, 2], [1, 0], [2, 4], [0, 1], [3, 3]], Operations: [[0, 'skip_connect'], [0, 'skip_connect'], [1, 'diff_gcn'], [1, 'skip_connect'], [2, 's_trans'], [0, 'dcc_1'], [3, 'trans']]"
#  "- Architecture: [[0, 4], [0, 4], [1, 4], [1, 1], [2, 3], [1, 3], [3, 4]], Operations: [[0, 's_trans'], [0, 's_trans'], [1, 's_trans'], [1, 'dcc_1'], [2, 'trans'], [1, 'trans'], [3, 's_trans']]"
#  "- Architecture: [[0, 1], [0, 3], [1, 2], [0, 3], [2, 3], [2, 4], [3, 4]], Operations: [[0, 'dcc_1'], [0, 'trans'], [1, 'diff_gcn'], [0, 'trans'], [2, 'trans'], [2, 's_trans'], [3, 's_trans']]"
#  "- Architecture: [[0, 3], [0, 4], [1, 3], [0, 0], [2, 2], [1, 1], [3, 1]], Operations: [[0, 'trans'], [0, 's_trans'], [1, 'trans'], [0, 'skip_connect'], [2, 'diff_gcn'], [1, 'dcc_1'], [3, 'dcc_1']]"
#  "- Architecture: [[0, 2], [0, 0], [1, 4], [0, 1], [2, 2], [1, 0], [3, 3]], Operations: [[0, 'diff_gcn'], [0, 'skip_connect'], [1, 's_trans'], [0, 'dcc_1'], [2, 'diff_gcn'], [1, 'skip_connect'], [3, 'trans']]"
#  "- Architecture: [[0, 4], [0, 1], [1, 2], [1, 2], [2, 3], [2, 1], [3, 1]], Operations: [[0, 's_trans'], [0, 'dcc_1'], [1, 'diff_gcn'], [1, 'diff_gcn'], [2, 'trans'], [2, 'dcc_1'], [3, 'dcc_1']]"
#  "- Architecture: [[0, 1], [0, 0], [1, 0], [0, 4], [2, 4], [2, 4], [3, 1]], Operations: [[0, 'dcc_1'], [0, 'skip_connect'], [1, 'skip_connect'], [0, 'trans'], [2, 's_trans'], [2, 's_trans'], [3, 'dcc_1']]"
#  "- Architecture: [[0, 1], [0, 3], [1, 2], [0, 0], [2, 4], [1, 2], [3, 3]], Operations: [[0, 'dcc_1'], [0, 'trans'], [1, 'diff_gcn'], [0, 'skip_connect'], [2, 's_trans'], [1, 'diff_gcn'], [3, 'trans']]"
#  "- Architecture: [[0, 2], [0, 4], [1, 1], [1, 1], [2, 3], [0, 4], [3, 4]], Operations: [[0, 'diff_gcn'], [0, 's_trans'], [1, 'dcc_1'], [1, 'dcc_1'], [2, 'trans'], [0, 's_trans'], [3, 's_trans']]"
#  "- Architecture: [[0, 2], [0, 4], [1, 3], [1, 0], [2, 2], [2, 2], [3, 2]], Operations: [[0, 'diff_gcn'], [0, 's_trans'], [1, 'trans'], [1, 'skip_connect'], [2, 'diff_gcn'], [2, 'diff_gcn'], [3, 'diff_gcn']]"
#  "- Architecture: [[0, 2], [0, 2], [1, 2], [0, 3], [2, 3], [1, 1], [3, 4]], Operations: [[0, 'diff_gcn'], [0, 'diff_gcn'], [1, 'diff_gcn'], [0, 'trans'], [2, 'trans'], [1, 'dcc_1'], [3, 's_trans']]"
#  "- Architecture: [[0, 2], [0, 1], [1, 1], [1, 4], [2, 2], [2, 0], [3, 2]], Operations: [[0, 'diff_gcn'], [0, 'dcc_1'], [1, 'dcc_1'], [1, 's_trans'], [2, 'diff_gcn'], [2, 'skip_connect'], [3, 'diff_gcn']]"
#  "- Architecture: [[0, 2], [0, 4], [1, 1], [0, 3], [2, 1], [0, 4], [3, 3]], Operations: [[0, 'diff_gcn'], [0, 's_trans'], [1, 'dcc_1'], [0, 'trans'], [2, 'dcc_1'], [0, 's_trans'], [3, 'trans']]"
#  "- Architecture: [[0, 0], [0, 0], [1, 2], [1, 4], [2, 1], [2, 4], [3, 2]], Operations: [[0, 'skip_connect'], [0, 'skip_connect'], [1, 'diff_gcn'], [1, 's_trans'], [2, 'dcc_1'], [2, 's_trans'], [3, 'diff_gcn']]"
#  "- Architecture: [[0, 4], [0, 0], [1, 3], [1, 2], [2, 4], [1, 4], [3, 2]], Operations: [[0, 's_trans'], [0, 'skip_connect'], [1, 'trans'], [1, 'diff_gcn'], [2, 's_trans'], [1, 's_trans'], [3, 'diff_gcn']]"
#  "- Architecture: [[0, 0], [0, 1], [1, 1], [0, 2], [2, 2], [1, 2], [3, 3]], Operations: [[0, 'skip_connect'], [0, 'dcc_1'], [1, 'dcc_1'], [0, 'diff_gcn'], [2, 'diff_gcn'], [1, 'diff_gcn'], [3, 'trans']]"
#  "- Architecture: [[0, 0], [0, 1], [1, 4], [1, 2], [2, 4], [0, 4], [3, 1]], Operations: [[0, 'skip_connect'], [0, 'dcc_1'], [1, 'dcc_1'], [1, 'diff_gcn'], [2, 's_trans'], [0, 's_trans'], [3, 'dcc_1']]"
#  "- Architecture: [[0, 3], [0, 3], [1, 2], [1, 0], [2, 0], [2, 1], [3, 1]], Operations: [[0, 'trans'], [0, 'trans'], [1, 'diff_gcn'], [1, 'skip_connect'], [2, 'trans'], [2, 'dcc_1'], [3, 'dcc_1']]"
#  "- Architecture: [[0, 4], [0, 4], [1, 2], [0, 2], [2, 4], [1, 3], [3, 4]], Operations: [[0, 's_trans'], [0, 's_trans'], [1, 'diff_gcn'], [0, 'diff_gcn'], [2, 's_trans'], [1, 'trans'], [3, 's_trans']]"
 "Based on the insights from similar benchmark datasets, consider the potential patterns or underlying principles in the top model designs. This includes commonalities in the choice of operations, preferences for certain macro architecture configurations, or any recurring themes that might indicate a successful approach to constructing spatiotemporal architectures for similar types of data. Evaluate these patterns and, using your comprehensive analysis, suggest an optimal model design for the source dataset. Consider how specific operations and architecture designs have contributed to high performance in similar datasets. Your suggestion should reflect a thoughtful synthesis of these insights, aiming to capture the most effective elements of the provided designs. Additionally, pay attention to the similarity scores between datasets, if provided, to gauge the relevance of each design's features in relation to the source dataset.\n")
                
            # prompt += ("Now, please provide a suggested architecture and set of operations for the source dataset, "
            #            "tailoring each recommendation to maximize potential performance based on the observed design "
            #            "patterns."                  "For example, Architecture: [[0, 2], [0, 3], [1, 1], [1, 2], [2, 2], [2, 1], [3, 0]], Operations: [[0, 'diff_gcn'], [0, 'trans'], [1, 'dcc_1'], [1, 'diff_gcn'], [2, 'diff_gcn'], [2, 'dcc_1'], [3, 'skip_connect']] "
            #       "means that there are four modules in a model, the first [0, 'diff_gcn'] represents the 1st module connecting the 0th module, and the operator used for the connection is 'diff_gcn' operator, "
            #       "the second [0, 'trans'] represents the 2nd module connecting the 0th module, The operator used is the 'trans' operator, [1, 'dcc_1'] represents the second module and connects the first module, "
            #       "the operator used is 'dcc_1', [1, 'diff_gcn'] represents the third module connects the first module, the operator used is 'diff_gcn', [2, 'diff_gcn'] means that the third module is also connected "
            #       "to the second module, using the operator 'diff_gcn', [2, 'dcc_1'] means that the fourth module is connected to the second module, using the operator 'dcc_1', [3, 'skip_connect'] means that the "
            #       "fourth module is connected to the third module, using the operator 'skip_connect'. ")
        else:
            prompt += ("Now, please provide a suggested architecture and set of operations for the source dataset, "
                       "tailoring each recommendation to maximize potential performance based on your knowledge.")
        prompt += ("Here are some suggestions from two model design experts for your model design. Please refer to them:\nExpert 1:")
        prompt += str(ar_suggestion)
        prompt += ("\nExpert 2:")
        prompt += str(model_suggestion)
        prompt += ("\nYour suggested optimal model design for the source dataset should be in the same search space we "
                   "defined. Your answer should be in the following format:\n")
        prompt += f"For the unseen dataset: (Architecture: [TBD])\nReasons:\n"
        print("intro + sim_dataset_prompt + prompt,",intro + sim_dataset_prompt + prompt)
        print("intro + prompt,",intro + prompt)
        return intro + sim_dataset_prompt + prompt , intro + prompt
    
    def generate_llm_mutation_prompt(self, dataset_name, promoted_child, current_design, generation, gnas_history, 
                                     best_design,  candidate_pools,detailed_log=None):
        """
        Generate a prompt for the LLM to help refine the promoted child model based on the provided GNAS context.

        :param dataset_name: Name of the dataset being optimized.
        :param promoted_child: The child model selected for further mutation.
        :param current_design: Current model design before the generation of children.
        :param generation: Current generation number in the evolutionary search.
        :param gnas_history: Historical record of all generations and their model performances.
        :param best_design: The best model design encountered so far in terms of performance.
        :param detailed_log: Flag to include detailed training logs in the prompt.
        :param candidate_pools: Information about top-performing designs from the most similar dataset.
        :return: A string prompt for the LLM.
        """
        intro = self.generate_GNAS_task_description() + self.generate_short_space_description()

        # Building the history narrative
        history = f"Currently, you are the evolutionary Spatiotemporal NAS agent at {generation} generation. We have already explored various Spatiotemporal Neural Network architectures to optimize performance. Your further recommendation should not repeat any of the models in the optimization trajectory (history) below:\n"

        # Iterate over the history dictionary, sorted by iteration keys to maintain order
        for iter_num in sorted(gnas_history.keys(), key=int):
            if iter_num == '0':
                detail_list = gnas_history[iter_num]
                history += f"Generation {iter_num} tested {len(detail_list)} children:\n"
                for details in detail_list:
                    architecture = details['link']
                    part1 = architecture[:1] 
                    part2 = architecture[1:3]  
                    part3 = architecture[3:5]  
                    part4 = architecture[5:]  

                    for i in range(3):
                        if not any(item[0] == i for item in part3):
                            part3.append([i, 5])  

                    for i in range(4):
                        if not any(item[0] == i for item in part4):
                            part4.append([i, 5])  
                    part3 = sorted(part3, key=lambda x: x[0])
                    part4 = sorted(part4, key=lambda x: x[0])
                    arch = part1 + part2 + part3 + part4
                    mapping = {
                        0: "skip_connect",
                        1: "gated_tcn",
                        2: "diff_gcn",
                        3: "trans",
                        4: "s_trans",
                        5: "none"
                    }
                    result = []
                    group = []  

                    for i in range(len(arch)):
                        part = arch[i]
                        first_element = part[0]  
                        second_element = part[1]  
                        action = mapping[second_element]
                        group.append(f"|{action}~{first_element}|")
                        if i + 1 < len(arch) and arch[i + 1][0] != 0:
                            result.append("+".join(group))
                            group = [] 

                    if group:
                        result.append("+".join(group))

                    final_result = "".join(result)
                    final_result = final_result.replace("||", "|")

                    history += f" - Achieved a performance of {round(details['perf'], 3)} with architecture {final_result}.\n"
            else:
                details = gnas_history[iter_num]
                history += f" - Generation {iter_num} achieved a performance of {round(details['perf'])} with operations {details['link']} which represents architecture {details['ops']}.\n"
        architecture = best_design['link']
        part1 = architecture[:1] 
        part2 = architecture[1:3]  
        part3 = architecture[3:5]  
        part4 = architecture[5:]  

        for i in range(3):
            if not any(item[0] == i for item in part3):
                part3.append([i, 5])  

        for i in range(4):
            if not any(item[0] == i for item in part4):
                part4.append([i, 5])  
        part3 = sorted(part3, key=lambda x: x[0])
        part4 = sorted(part4, key=lambda x: x[0])
        arch = part1 + part2 + part3 + part4
        mapping = {
            0: "skip_connect",
            1: "gated_tcn",
            2: "diff_gcn",
            3: "trans",
            4: "s_trans",
            5: "none"
        }
        result = []
        group = []  

        for i in range(len(arch)):
            part = arch[i]
            first_element = part[0]  
            second_element = part[1]  
            action = mapping[second_element]
            group.append(f"|{action}~{first_element}|")
            if i + 1 < len(arch) and arch[i + 1][0] != 0:
                result.append("+".join(group))
                group = [] 

        if group:
            result.append("+".join(group))
        final_best_result = ""
        final_best_result = "".join(result)
        final_best_result = final_best_result.replace("||", "|")
        # Highlighting the best model so far
        history += f"The best model design so far is architecture {final_best_result}, which achieved MAE of {round(best_design['perf'])} at generation {best_design['iteration']}.\n"

        # If detailed logs are available, add them to the prompt
        log = ""
        if detailed_log:
           log = f"Here is the training log snapshot (every 25 epochs) of the last experiment with operations {current_design['ops']} and macro architecture {current_design['link']}:\n"
           for log_entry in current_design["detailed_log"]:
               log += f"Epoch {log_entry['epoch']}: Train Acc - {log_entry['train_accuracy']}, Val Acc - {log_entry['val_accuracy']}, Test Acc - {log_entry['test_accuracy']}, Train Loss - {log_entry['train_loss']}, Val Loss - {log_entry['val_loss']}, Test Loss - {log_entry['test_loss']};\n"

        # Include insights from candidate pools if available
        if candidate_pools:
            top_models = candidate_pools[dataset_name][0]
            print("top_models:",top_models)
            selected_dataset = top_models['selected_dataset']
            top_models = top_models['top_models']
            knowledge = f"\nAdditionally, insights into top-performing designs in the most similar benchmark dataset {selected_dataset} include:\n"

            for model in top_models:
                architecture = model[0]
                part1 = architecture[:1] 
                part2 = architecture[1:3]  
                part3 = architecture[3:5]  
                part4 = architecture[5:]  

                for i in range(3):
                    if not any(item[0] == i for item in part3):
                        part3.append([i, 5])  

                for i in range(4):
                    if not any(item[0] == i for item in part4):
                        part4.append([i, 5])  
                part3 = sorted(part3, key=lambda x: x[0])
                part4 = sorted(part4, key=lambda x: x[0])
                arch = part1 + part2 + part3 + part4
                mapping = {
                    0: "skip_connect",
                    1: "gated_tcn",
                    2: "diff_gcn",
                    3: "trans",
                    4: "s_trans",
                    5: "none"
                }
                result = []
                group = []  

                for i in range(len(arch)):
                    part = arch[i]
                    first_element = part[0]  
                    second_element = part[1]  
                    action = mapping[second_element]
                    group.append(f"|{action}~{first_element}|")
                    if i + 1 < len(arch) and arch[i + 1][0] != 0:
                        result.append("+".join(group))
                        group = [] 

                if group:
                    result.append("+".join(group))
                final_model_result = ""
                final_model_result = "".join(result)
                final_model_result = final_model_result.replace("||", "|")
                knowledge += f" - (Architecture: {final_model_result})\n"
        else:
            knowledge = ""

        # Children details for selection
        #prompt += "We have completed the crossover on the best child from the last generation with respect to the top " \
        #         "model designs from the second and third similar datasets. Here is the current generation of " \
        #         "children for selection:\n"
        instruction = f"\nWe have completed the exploration (crossover) on the best design so far with respect to the top model designs from the second and third similar datasets. Here are the kids who had the highest performance boost (with the lowest MAE value) on the most similar dataset:\n"
        architecture = promoted_child['link']
        part1 = architecture[:1] 
        part2 = architecture[1:3]  
        part3 = architecture[3:5]  
        part4 = architecture[5:]  

        for i in range(3):
            if not any(item[0] == i for item in part3):
                part3.append([i, 5])  

        for i in range(4):
            if not any(item[0] == i for item in part4):
                part4.append([i, 5])  
        part3 = sorted(part3, key=lambda x: x[0])
        part4 = sorted(part4, key=lambda x: x[0])
        arch = part1 + part2 + part3 + part4
        mapping = {
            0: "skip_connect",
            1: "gated_tcn",
            2: "diff_gcn",
            3: "trans",
            4: "s_trans",
            5: "none"
        }
        result = []
        group = []  

        for i in range(len(arch)):
            part = arch[i]
            first_element = part[0]  
            second_element = part[1]  
            action = mapping[second_element]
            group.append(f"|{action}~{first_element}|")
            if i + 1 < len(arch) and arch[i + 1][0] != 0:
                result.append("+".join(group))
                group = [] 

        if group:
            result.append("+".join(group))
        final_promoted_result = ""
        final_promoted_result = "".join(result)
        final_promoted_result = final_promoted_result.replace("||", "|")
        instruction += f"Promoted Child: Architecture {final_promoted_result}\n"
        # Finally, ask for suggestions on improvements
        instruction += f"\nAs an optimal Spatiotemporal NAS that performs exploitation (mutation), please further refine (mutate) this promoted child ({final_promoted_result}) based on its potential effectiveness, the experiment history"
        #prompt += "\nAs an optimal Graph NAS, please suggest the best child from the current generation for further " \
        #          "validation based on their potential effectiveness, the history of experimental performances"
        if detailed_log and candidate_pools:
            instruction += ", training log of last trial, and the potential pattern of top-performing designs in the most similar datasets."
        elif detailed_log:
            instruction += " and training log of last trial."
        elif candidate_pools:
            instruction += " and the potential pattern of top-performing designs in the most similar datasets."
        else:
            instruction += "."
        instruction += "The objective is to maximize the model's performance so that the MAE value of the model for downstream task testing is as low as possible. You should modify upon the promoted child and shouldn't repropose model designs that have already been validated in the optimization trajectory. Your suggested optimal model design for the unseen dataset should be in the same search space we defined and should not repeat any model design already contained in the experiment history. Your answer should closely follow the output format below:\n\n"
        instruction += f"Response Format:\nFor the unseen dataset, I recommend (Architecture: [TBD]).\nReasons for recommendation: TBD\n"

        #return intro + history + log + knowledge + instruction
        return intro + history  + knowledge + instruction
    
    @staticmethod
    def generate_GNAS_task_description():
        return "You are Quoc V. Le, a computer scientist and artificial intelligence researcher who is widely regarded as one of the leading experts in deep learning and neural network architecture search. Your work in this area has focused on developing efficient algorithms for searching the space of possible neural network architectures, with the goal of finding architectures that perform well on a given task while minimizing the computational cost of training and inference. Your task is to assist me in selecting the best operations to design a neural network block using the available operations, aiming to perform the neural architecture search of the Spatiotemporal Neural Network on the unseen spatiotemporal dataset. The objective is to maximize the model's performance and minimize the MAE value of the model."

    @staticmethod
    def generate_short_space_description():
        return "\nIn the context of spatiotemporal neural networks, the 6 available operations are as follows:\n- 'skip_connect': Skip Connection, a mechanism to bypass intermediate layers, preserving original input information and supporting residual learning.    \n- 'gated_tcn': Dynamic Convolutional Cell, a convolutional unit that adapts to varying graph structures for flexible and robust feature learning.    \n- 'diff_gcn': Diffusion Graph Convolutional Network, employs diffusion processes to spread stage features across the graph, capturing long-range dependencies.   \n- 'trans': Transformer, uses self-attention to model relationships between stages, effective for capturing global interactions.   \n- 's_trans': Simplified Transformer, a lightweight version of the transformer model, optimized for faster global feature extraction in graphs.   \n- 'none': This operation indicates that there is no operational connection between the two stages, it breaks the gradient flow between two stages.   In the context of spatiotemporal neural networks, the design of a model is described by two main components:\nThe neural network block is defined by 10 operations (i.e., op_list = [op0, op1, op2, op3, op4, op5, op6, op7, op8, op9]), which represent the operations executed between various stages of the block.     This block comprises 5 stages, labeled as s0, s1, s2, s3 and s4, each corresponding to distinct feature maps in the neural network.   s0 serves as the input feature map for this block.   \ns1 will be calculated by s1 = op0(s0).   \ns2 will be calculated by s2 = op1(s0) + op2(s1).   \ns3 will be calculated by s3 = op3(s0) + op4(s1) + op5(s2).   \ns4 will be calculated by s4 = op6(s0) + op7(s1) + op8(s2) + op9(s3).   \nNote that s4 becomes the output for this block and serves as the input for the subsequent block.\n\n"#"To recall, in the context of Spatiotemporal Neural Network, the design of a model is described by two main components:\n1. The macro architecture list defines how the operations are connected in a directed acyclic graph (DAG). It is specified as a nested list of seven sublist. The first element of each of these sublists represents the preorder module sequence number of the current module connection, and the second element represents the operation or operation ID used for the connection. The following are the rules for module connection:(1). Every module m (except module 0 and module 1) must be connected to two different preorder modules , which guarantees the structure of a directed acyclic graph. (2). Modules 0 ,as initial modules, cannot be connected and therefore have no input, only output. Module 1 must be connected to module 0.\n2. The operation list consists of a set of operations that can be used in constructing a Spatiotemporal Neural Network. We consider 5 candidate operations of Spatiotemporal Neural Network, which are: 'skip_connect', 'dcc_1','diff_gcn', 'trans', 's_trans.' \nTogether, these components define the computation graph of a Spatiotemporal Neural Network, including the flow of data through various operations. You need to understand the real structure of the Spatiotemporal Neural Networks given its operation list and macro architecture list.\n"

    @staticmethod
    def extract_model_designs(llm_response, dataset_name):
        """
        Extracts model designs suggested by the LLM for each source dataset.

        :param llm_response: A string containing the LLM's response in the specified format.
        :return: A dictionary with source dataset names as keys and their suggested model designs as values.
        """
        # Pattern to match the format of the LLM's response for each dataset
        #pattern = r"\(Architecture: (\[.*?\]), Operations: (\[.*?\])\)"
        pattern = r"\(Architecture:.*?\)"
        # Find all matches in the response
        matches = re.findall(pattern, llm_response)

        # Initialize a dictionary to hold the extracted designs
        suggested_designs = {}

        # Iterate through all matches and populate the dictionary
        for match in matches:
            architecture = match
            mapping = {
                    'skip_connect': 0,
                    'gated_tcn': 1,
                    'diff_gcn': 2,
                    'trans': 3,
                    's_trans': 4,
                    'none': 5
                }
            architecture = architecture.replace("|", "||")
            pattern = r"\|([a-zA-Z_0-9]+)~(\d)\|"
            matches = re.findall(pattern, architecture)
            result = []
            for match in matches:
                operation, num = match
                operation_num = mapping[operation]
                if operation_num != 5:
                    result.append([int(num), operation_num])
            operations = [[num, list(mapping.keys())[list(mapping.values()).index(op)]] for num, op in result]
            print("result:",result)
            print("operations:",operations)
            suggested_designs[dataset_name] = {
                "link": eval(str(result)),  # Use eval to convert string representation of list to actual list
                "ops": eval(str(operations))
            }

        return suggested_designs

    @staticmethod
    def has_bad_models(models_info):
        for model in models_info:
            if 'bad_models' in model and model['bad_models']:
                return True
        return False

