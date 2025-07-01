# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license

import random
from langchain_core.utils.function_calling import convert_to_openai_function
from sympy import N
from .NASBenchGraphCTS import run_CTS_experiment
import copy
import re
from langchain_community.chat_models import ChatZhipuAI
class GNASPipeline:
    def __init__(self, search_strategy, llm_prompt_configurator, gnn_benchmark, langchain_query, file_path, use_parser,
                 candidate_pools=None, max_iter=10, n=1, num_children=1):
        """
        Initializes the Graph NAS process.
        :param search_strategy: Search strategy used in the Graph NAS process.
        :param candidate_pools: Candidate pool of model designs.
        """
        self.search_strategy = search_strategy
        self.langchain_query = langchain_query
        self.candidate_pools = candidate_pools
        self.use_training_log = False
        self.gnn_benchmark = gnn_benchmark
        self.llm_prompt_configurator = llm_prompt_configurator
        self.file_path = file_path
        self.use_parser = use_parser
        self.max_iter = max_iter
        self.n = n
        self.num_children = num_children
        self.benchmarking = False

    def is_valid_refined_child(refined_child):
    # 正则表达式用于匹配指定格式
        pattern = r'^\[\[(\d+,\d+)(,\s*(\d+,\d+))*\]$'
    # 检查每对 (n, m)，确保 m 在 0 到 4 之间
        pairs = re.findall(r'\[(\d+),(\d+)\]', refined_child)
        for n, m in pairs:
            if not (0 <= int(m) <= 4):
                return False
        return True
    
    def run_gnas(self, dataset_name, initial_detailed_infos_list, selected_operator,architecture_rank,initial_prompt,cuda_number,benchmarking=False, llm_no_candidates=False):
        self.benchmarking = benchmarking
        if 'designn' in self.search_strategy:
            return self.designn_search(dataset_name, initial_detailed_infos_list, selected_operator,architecture_rank,initial_prompt,llm_no_candidates,cuda_number)
        else:
            raise ValueError("Unsupported strategy specified.")
        

    def designn_search(self, dataset_name, initial_detailed_infos_list, selected_operator,architecture_rank,initial_prompt,llm_no_candidates,cuda_number):
        """
        Perform the Model Proposal Refinement of DesiGNN.

        :param dataset_name: Name of the dataset being tested.
        :param dataloader: DataLoader providing the dataset for training and validation.
        :param initial_detailed_infos: Dictionary containing initial model design and its performance.
        :param llm_no_candidates: Flag indicating whether the LLM uses knowledge to refine the design.
        """
        n_initial = len(initial_detailed_infos_list)
        if n_initial < self.n:
            raise ValueError(f"Number of initial designs ({n_initial}) is less than the required number of designs ({self.n}).")

        initial_detailed_infos = min(initial_detailed_infos_list, key=lambda x: x['perf'])
        current_design = initial_detailed_infos
        best_performance = initial_detailed_infos['perf']
        best_design = initial_detailed_infos
        best_design['iteration'] = 0
        gnas_history = {
            '0': []
        }
        for i in range(n_initial):
            gnas_history['0'].append({
                'perf': initial_detailed_infos_list[i]['perf'],
                'link': initial_detailed_infos_list[i]['link'],
                'ops': initial_detailed_infos_list[i]['ops'],
                'best': best_performance,
                'promoted': None
            })

        merged_pool = []
        for similar_dataset in list(self.candidate_pools.values())[0][0:n_initial]:
            merged_pool.extend(similar_dataset['top_models'])
        
        # Evolutionary search through generations
        top1_knowledge = self.candidate_pools[dataset_name][0]['selected_dataset']
        last_promoted = None
        for generation in range(self.max_iter):
            while True:
                children = []
                # Exploration: Generate new models using mutation and crossover from candidate pools
                for _ in range(self.num_children):
                    child = self.controlled_exploration(best_design, merged_pool)
                    children.append(child)
                estimated_performances = self.gnn_benchmark.extract_performances(top1_knowledge, children)
                promoted_child = children[estimated_performances.index(min(estimated_performances))]

                if last_promoted is None or promoted_child != last_promoted:
                    last_promoted = promoted_child
                    break
            
            promoted_child_performance = None
            if self.benchmarking:
                details = self.gnn_benchmark.extract_single_performance(dataset_name, {dataset_name: promoted_child})
                promoted_child_performance = details['perf']
                print(f"Generation {generation + 1}: Promoted child: {promoted_child['link']} {promoted_child['ops']} Performance: {promoted_child_performance}")

            # Construct prompt to let LLM select the most promising child
            if self.use_parser:
                raise NotImplementedError("Parser not supported for this search strategy.")
            else:
                knowledge = self.candidate_pools if not llm_no_candidates else None
                prompt = self.llm_prompt_configurator.generate_llm_mutation_prompt (dataset_name, promoted_child, 
                                                                                   current_design, generation + 1, gnas_history, best_design, knowledge,
                                                                                   self.use_training_log )
                print("prompt:",prompt)
                while True:
                    refined_child = self.query_llm_for_directional_exploitation(prompt, generation + 1, dataset_name)
                    pattern = r'\[\s*(\[\s*\d+\s*,\s*[0-4]\s*\]\s*,\s*){6}\[\s*\d+\s*,\s*[0-4]\s*\]\s*\]'
                    link_value = str(refined_child[dataset_name]["link"])
                    if re.match(pattern, link_value) :
                        pairs = re.findall(r'\[(\d+),(\d+)\]', link_value)
                        if all(0 <= int(m) <= 4 for _, m in pairs):
                            break                         

                print("refined_child:",refined_child)
            # Evaluate the selected child using the model training and validation function
            if self.benchmarking:
                new_detailed_infos = self.gnn_benchmark.extract_single_performance(dataset_name, refined_child)
                new_detailed_infos['detailed_log'] = self.gnn_benchmark.extract_single_log(dataset_name, refined_child)
            else:
                print(f'refined_child[{dataset_name}]["ops"]: {refined_child[dataset_name]["ops"]}')
                print(f'refined_child[{dataset_name}]["link"]: {refined_child[dataset_name]["link"]}')
                new_detailed_infos = run_gnn_experiment(dataset_name, refined_child[dataset_name]["link"],
                                                        refined_child[dataset_name]["ops"],cuda_number)
            performance = new_detailed_infos['perf']
            if performance < best_performance:
                best_design = new_detailed_infos
                best_design['iteration'] = generation + 1
                best_performance = performance
            print(f"Generation {generation + 1}: Suggested new model design {refined_child[dataset_name]['link']} {refined_child[dataset_name]['ops']} Performance: {performance}")

            #评估agent
            # langchain_query = ChatZhipuAI(temperature=1, model='glm-4-plus')
            # prompt=""
            # response = langchain_query.invoke(prompt)

            # Update current design with the new suggested design
            gnas_history[str(generation + 1)] = {
                'perf': new_detailed_infos['perf'],
                'link': new_detailed_infos['link'],
                'ops': new_detailed_infos['ops'],
                'best': best_performance,
                'promoted': {
                    'link': promoted_child['link'],
                    'ops': promoted_child['ops'],
                    'perf': promoted_child_performance
                }
            }
            current_design = new_detailed_infos
            generation += 1

        return best_design, gnas_history

    
    def controlled_exploration(self, current_design, merged_pool):
        """
        Generate a new child model by performing a controlled crossover between the current best design 
        and a randomly selected model from the merged pool.

        :param current_design: The current best design, typically from previous iterations.
        :param merged_pool: A list containing the top models from the two most similar datasets.
        :return: A dictionary representing the child model with new 'link' (architecture) and 'ops' (operations).
        """
        # Randomly select a model from the merged pool for crossover
        random_model = random.choice(merged_pool)
        
        # 1. Perform single-point crossover with adaptive rolling.
        op_id = [0, 1, 2, 3, 4]
        first_part_a, second_part_a = current_design['link'][:3], current_design['link'][3:]
        first_part_b, second_part_b = random_model[0][:3], random_model[0][3:]
        candidates = [[[0, v1], [0, v2], [1, v3]] for v1 in op_id for v2 in op_id for v3 in op_id]
        overlap = [sp for sp in candidates if sp == first_part_a or sp == first_part_b]
        first_part_child = random.choice(overlap)

        # Get possible second parts based on the selected first part
        # overlap = [second_part_a,second_part_b]
        # second_part_child = random.choice(overlap)
        candidates = self.second_part_candidates(first_part_child)
        overlap = [sp for sp in candidates if sp == second_part_a or sp == second_part_b]
        if overlap:
            second_part_child = random.choice(overlap)
        else:
            second_part_child = random.choice(candidates)

        # Combine first and second parts to form the child
        new_architecture = first_part_child + second_part_child

        # 2. Introduce slight changes to a promising operation list based on another example.
        current_design_ops = [sublist[1] for sublist in current_design['ops']]
        random_model_ops = [sublist[1] for sublist in random_model[1]]
        differences = [i for i, (a, b) in enumerate(zip(current_design_ops, random_model_ops)) if a != b]

        #differences = [i for i, (a, b) in enumerate(zip(current_design['ops'], random_model[1])) if a != b]
    
        # Decide on the number of changes; here we use 1 or 2 changes for 'slight' modification
        if new_architecture == current_design['link']:
            num_changes = random.choice([1, 2, 3, 4, 5, 6, 7]) if len(differences) > 1 else 1
        else:
            num_changes = random.choice([0, 1, 2, 3, 4, 5, 6]) if len(differences) > 1 else 1
        
        # Select random differences to change
        change_points = random.sample(differences, min(num_changes, len(differences)))
        
        # Create a copy of the promising list to modify
        #new_operations = current_design['ops'][:]
        new_operations = current_design_ops[:]  #['diff_gcn', 'trans', 's_trans', 'diff_gcn', 'trans', 'dcc_1', 'skip_connect']
        
        # Introduce changes at the selected points
        for point in change_points:
            #new_operations[point] = random_model[1][point]
            new_operations[point] = random_model_ops[point]
        for i in range(len(new_architecture)):
            new_architecture[i][1] = new_operations[i]

        new_operations = copy.deepcopy(new_architecture)
        operation_names = [
           'skip_connect',  # 0
           'gated_tcn',         # 1
           'diff_gcn',      # 2
           'trans',         # 3
           's_trans'        # 4
        ]
       # 将字符串替换为对应的数字
        for sublist in new_operations:
            sublist[1] = operation_names.index(sublist[1])
        return {'link': new_operations, 'ops': new_architecture}
    
    @staticmethod
    def second_part_candidates(first_part_child):
        op_id = [0, 1, 2, 3, 4]
        """ Return valid second parts based on the first part of the structure. """
        return [[[0, v1], [1, v2], [0, v3], [1, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] + [
                [[0, v1], [1, v2], [0, v3], [2, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] + [
                [[0, v1], [1, v2], [0, v3], [3, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] + [
                [[0, v1], [1, v2], [1, v3], [2, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] + [
                [[0, v1], [1, v2], [1, v3], [3, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] + [
                [[0, v1], [1, v2], [2, v3], [3, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] + [
                [[0, v1], [2, v2], [0, v3], [1, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] + [
                [[0, v1], [2, v2], [0, v3], [2, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] + [
                [[0, v1], [2, v2], [0, v3], [3, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] + [
                [[0, v1], [2, v2], [1, v3], [2, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] + [
                [[0, v1], [2, v2], [1, v3], [3, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] + [
                [[0, v1], [2, v2], [2, v3], [3, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] + [
                [[1, v1], [2, v2], [0, v3], [1, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] + [
                [[1, v1], [2, v2], [0, v3], [2, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] + [
                [[1, v1], [2, v2], [0, v3], [3, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] + [
                [[1, v1], [2, v2], [1, v3], [2, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] + [
                [[1, v1], [2, v2], [1, v3], [3, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] + [
                [[1, v1], [2, v2], [2, v3], [3, v4]] for v1 in op_id for v2 in op_id for v3 in op_id for v4 in op_id] 
        return []  # Return an empty list if the first part is not recognized

    def query_llm_for_directional_exploitation(self, prompt, generation, dataset_name):
        
        patterns = [
            r"\(Architecture:.*?\)"
            #r"\(Architecture: (\[.*?\]), Operations: (\[.*?\])\)",
            #r"-\s*\*\*Architecture:\*\*\s*(\[.*?\])\s*-\s*\*\*Operations:\*\*\s*(\[.*?\])",
            #r"-\s*\*\*Architecture\*\*:\s*(\[.*?\])\s*-\s*\*\*Operations\*\*:\s*(\[.*?\])",
            #r"-\s*\*\*Architecture:\s*(\[.*?\])\*\*\s*-\s*\*\*Operations:\s*(\[.*?\])\*\*"
        ]
        #max_attempts = 5 
        #attempts = 0
        #while attempts < max_attempts:
        while True:
            try:
                refined_design = self.langchain_query.invoke(prompt, timeout=120)
                print("refined_design:",refined_design.content)
                matches = []
                for pattern in patterns:
                    matches.extend(re.findall(pattern, refined_design.content))
                    #matches = re.findall(pattern, refined_design.content)
                if matches:
                    break
                print("try again...")
            except TimeoutError:
                print(f"Timeout for generation {generation}.")
            #attempts += 1
        #if attempts == max_attempts:
            #print("Exceeded maximum attempts. Please check the request or input.")

            
        # Append the response to the file
        with open(self.file_path, 'a') as file:  # Open in append mode
            file.write(f"\nResponse for generation {generation}:\n")
            file.write(refined_design.content + "\n")
        children = self.llm_prompt_configurator.extract_model_designs(refined_design.content, dataset_name)

        return children
    
    # def is_valid_refined_child(refined_child):
    # # 正则表达式用于匹配指定格式
    #     pattern = r'^\[\[(\d+,\d+)(,\s*(\d+,\d+))*\]$'
    # # 检查每对 (n, m)，确保 m 在 0 到 4 之间
    #     pairs = re.findall(r'\[(\d+),(\d+)\]', refined_child)
    #     for n, m in pairs:
    #         if not (0 <= int(m) <= 4):
    #             return False
    #     return True

