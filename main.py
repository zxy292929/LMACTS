# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license

from LMACTS.DatasetUnderstanding.DatasetUnderstanding import GraphDatasetUnderstanding
from DatasetUnderstanding.DatasetReader import DatasetReader
from DatasetComparison.DatasetComparison import DatasetComparison
from LMACTS.LLMConfiguredAutoCTS.LLMConfiguredAutoCTS import LLMConfiguredAutoGNN
from MultiAgent.OperatorSelect import OperatorsSelect
from MultiAgent.ConnectionRule import ConnectionRule
import sys
import os
import time
import json
import argparse
from datetime import datetime
import torch
#from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI
os.environ["ZHIPUAI_API_KEY"] = ""
os.chdir(sys.path[0])
def main():
    parser = argparse.ArgumentParser(description="DesiGNN Framework")

    parser.add_argument('--dataset', type=str, default='PEMS07',
                        help='Dataset name. Options: ETTh1, ETTh2, ETTm1, ETTm2, exchange_rate, METR-LA, PEMS-BAY, PEMS03, PEMS04, PEMS07, PEMS08, solar.')
    parser.add_argument('--initialization', type=str, default='transfer',
                        help='Initialization method. Options: naive, transfer, none. Default: transfer.')
    parser.add_argument('--search_strategy', type=str, default='designn',
                        help='Default: designn.')
    

    parser.add_argument('--max_iter', type=int, default=10,
                        help='Maximum number of iterations for the DesiGNN. Default 30.')
    parser.add_argument('--k', type=int, default=30,
                        help='Number of top-performing models to consider. Default 30.')
    parser.add_argument('--g', type=int, default=20,
                        help='Number of graph topological features to consider in the Graph Dataset Comparison. Default 8.')
    parser.add_argument('--s', type=int, default=1,
                        help='Number of similar datasets to consider in the Initial Trial Suggestion. Default 3.')
    parser.add_argument('--n', type=int, default=3,
                        help='Number of similar datasets to consider in the controlled exploration. Default 3.')
    parser.add_argument('--n_children', type=int, default=30,
                        help='Number of children to generate in the DesiGNN. Default 30.')
    parser.add_argument('--use_bad_designs', action='store_true', default=False,
                        help='Also use bad designs as transferable knowledge. Default False.')
    parser.add_argument('--use_semantic', action='store_true', default=False,
                        help='Use semantic description in the Graph Dataset Comparison. Default False.')
    parser.add_argument('--no_statistics', action='store_true', default=False,
                        help='Do not use graph topological features in the Graph Dataset Comparison. Default False.')
    parser.add_argument('--no_reorder', action='store_true', default=False,
                        help='Do not reorder the candidate pools in the DesiGNN. Default False.')
    parser.add_argument('--llm_no_candidates', action='store_true', default=False,
                        help='Do not use candidates in the DesiGNN. Default False.')
    parser.add_argument('--use_parser', action='store_true', default=False,
                        help='Use output parser. Default False.')
    parser.add_argument('--cuda_number',type=str, default="cuda:7")
    # Valid only if initialization is naive
    parser.add_argument('--add_statistics', action='store_true', default=False,
                        help='Use semantic description in the Graph Dataset Comparison. Default False.')

    # Utilities

    parser.add_argument('--benchmarking', action='store_true', default=False,
                        help='Use benchmarking datasets for fast experiment. Default False.')
    parser.add_argument('--force_benchmark', type=str, default=None,
                        help='Force to borrow knowledge from a specific benchmark dataset. Default None.')
    args = parser.parse_args()

    # File path
    unseen_dataset_name = args.dataset
    print("unseen_dataset_name:",unseen_dataset_name)
    parts = unseen_dataset_name.split(':')
    short_name = parts[1] if len(parts) > 1 else unseen_dataset_name

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    response_save_path = os.path.join(os.getcwd(), 'responses', f"{short_name}", 
                                      f'{short_name}_{args.search_strategy}_{current_time}'.lower())
    os.makedirs(response_save_path, exist_ok=True)

    benchmark_datasets = ["ETTh1", "ETTm1", "exchange_rate", "METR-LA", "PEMS03", "PEMS04", "PEMS08", "solar"]
    #benchmark_datasets = ["ETTh1", "ETTh2", "ETTm1", "PEMS-BAY", "PEMS03", "PEMS08", "solar"]
    benchmark_datasets = [dataset for dataset in benchmark_datasets if args.dataset not in dataset]
    

    benchmarking = unseen_dataset_name in benchmark_datasets and args.benchmarking
    unseen_dir = 'datasets/Unseen datasets'
    benchmark_dir = 'datasets/Benchmark datasets'

    langchain_query = None
    latency1, latency2, latency3, latency4 = 0, 0, 0, 0

    t0 = time.time()


    metrics_list = []
    if not args.no_statistics:
        #metrics_list =['average_clustering_coefficient', 'local_average_betweenness_centrality', 'density', 'average_degree_centrality', 'local_average_closeness_centrality', 'average_degree', 'edge_count', 'local_graph_diameter', 'local_average_shortest_path_length', 'assortativity', 'average_eigenvector_centrality', 'feature_dimensionality', 'node_count', 'node_feature_diversity', 'connected_components', 'label_homophily']
        metrics_list =['local_temporal_granularity', 'local_time_span', 'local_cyclic_patterns', 'local_cycle_length', 'sum_values', 'abs_energy', 'mean_abs_change', 'mean_change', 'mean_second_derivative_central', 'median', 'mean', 'length', 'standard_deviation', 'variation_coefficient', 'variance', 'skewness', 'kurtosis', 'root_mean_square', 'absolute_sum_of_changes', 'longest_strike_below_mean', 'spatial_autocorrelation', 'temporal_autocorrelation', 'spatial_proximity', 'temporal_trends', 'spatial_and_temporal_resolution', 'spatial_lags', 'temporal_lags', 'spatio-temporal_clustering', 'directionality', 'spatial_variability', 'temporal_variability', 'spatial-temporal_dependency', 'mobility_patterns', 'spatio-temporal_interactions', 'hotspots', 'spatio-temporal_correlation', 'geographical_influence', 'periodicity']
        metrics_list = metrics_list[:args.g]

    user_description_path = f"datasets/Unseen datasets/{unseen_dataset_name}/user_description.txt"
    if os.path.exists(user_description_path):
        with open(user_description_path, 'r') as file:
            user_description = file.read()
    else:

        user_description = "The user does not provide a description for the dataset, please understand the dataset " \
                           "based entirely on the following spatio-temporal data features."

    print("user_description:",user_description)
    # Process unseen dataset
    understanding_module = GraphDatasetUnderstanding(unseen_dataset_name,
                                                     user_description=user_description,
                                                     metrics_list=metrics_list,
                                                     root_dir=unseen_dir,
                                                     no_statistics=args.no_statistics,
                                                     use_semantic=args.use_semantic,
                                                     num_samples=20,
                                                     num_hops=2,
                                                     seed=42)
    unseen_dataset_desc = understanding_module.process()
    print("unseen_dataset_desc:",unseen_dataset_desc)
    k = args.k
    mode = 'both' if args.use_bad_designs else 'best'
    if benchmarking:
        # Use benchmarking datasets for fast experiment+
        if args.s > 3:
            raise ValueError("The number of similar datasets should be less than or equal to 3 when using benchmarking command.")
        print("Benchmarking mode enabled. All the graph understanding and comparison hyperparameters are set to default.")

        with open('./LLMConfiguredAutoGNN/initial_model_benchmark.json', 'r') as file:
            initial_model_benchmark = json.load(file)
        most_similar = {args.dataset: initial_model_benchmark[unseen_dataset_name]['benchmark'][:args.s]}
        similarity_scores = None
    
        suggested_design_dict_list = []
        for similar_dataset in most_similar[unseen_dataset_name]:
            suggested_design_dict_list.append({unseen_dataset_name: initial_model_benchmark[unseen_dataset_name][similar_dataset]})

        print(f'Most similar: {most_similar}')
        print(f'Suggested design dict: {suggested_design_dict_list}\n')

        llm_configured_autognn = LLMConfiguredAutoGNN(most_similar, use_parser=args.use_parser, k=k, mode=mode)
        candidate_pools = llm_configured_autognn.generate_candidate_pools()
    else:
        if args.initialization == 'transfer':

            langchain_query = ChatZhipuAI(temperature=1, model='glm-4-plus')

            if args.force_benchmark:
                print(f"Forced transfer of {args.force_benchmark} knowledge to the unseen dataset.")
                #if args.force_benchmark not in ["Cora", "CiteSeer", "PubMed", "CS","Physics", "Photo", "Computers", 
                #                                "ogbn-arxiv", "ogbn-proteins"]:
                if args.force_benchmark not in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange_rate", "METR-LA", "PEMS-BAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08", "solar"]:
                    raise ValueError("Forced benchmark dataset not found in the benchmark datasets.")
                most_similar = {args.dataset: [args.force_benchmark]}
                similarity_scores = None
                print(f'most_similar: {most_similar}')
            else:
                # Process benchmark datasets
                benchmark_dataset_desc = {}
                for benchmark_dataset_name in benchmark_datasets:
                    benchmark_dataset_understanding = GraphDatasetUnderstanding(benchmark_dataset_name,
                                                                                user_description=None,
                                                                                metrics_list=metrics_list,
                                                                                root_dir=benchmark_dir,
                                                                                no_statistics=args.no_statistics,
                                                                                use_semantic=args.use_semantic,
                                                                                num_samples=20,
                                                                                num_hops=2,
                                                                                seed=42)
                    benchmark_dataset_desc[benchmark_dataset_name] = benchmark_dataset_understanding.process()
                print("benchmark_dataset_desc:",benchmark_dataset_desc)
                latency1 = time.time() - t0
                print(f"Graph Understanding Module Latency: {latency1}")

                t0 = time.time()
                dataset_comparison = GraphDatasetComparison(unseen_dataset_description=unseen_dataset_desc,
                                                            benchmark_dataset_descriptions=benchmark_dataset_desc,
                                                            unseen_dataset_name=unseen_dataset_name,
                                                            benchmark_datasets=benchmark_datasets,
                                                            unseen_dir=unseen_dir,
                                                            benchmark_dir=benchmark_dir,
                                                            use_parser=True)

                task_similarity_file_name = os.path.join(response_save_path, 
                                                         f"1_task_similarity_response_{current_time}.txt")
                textual_similarity_response = dataset_comparison.compare_datasets(langchain_query, 
                                                                                  task_similarity_file_name)
                                                                                  
                print("textual_similarity_response:",textual_similarity_response)
                print("unseen_dataset_name:",unseen_dataset_name)
                n = max(args.s, args.n)

                most_similar, similarity_scores = dataset_comparison.analyze_similarity_scores_from_dict(
                    textual_similarity_response, unseen_dataset_name, n)
                latency2 = time.time() - t0
                print(f"Graph Dataset Comparison Latency: {latency2}")
                print(f'Similarity scores: {similarity_scores}')

            ori_selected_operator,memory_op=OperatorsSelect.OperatorsSelect(unseen_dataset_desc,similarity_scores)
            ori_selected_architecture,memory_arch=ConnectionRule.ConnectionRule(unseen_dataset_desc,similarity_scores)
            architecture_suggestion=OperatorsSelect.Suggestions(ori_selected_operator,memory_op)
            operator_suggestion=ConnectionRule.Suggestions(ori_selected_architecture,memory_arch)
            print("architecture_suggestion:",architecture_suggestion)
            print("operator_suggestion:",operator_suggestion)
            selected_operator=OperatorsSelect.ModifiedOperator(operator_suggestion,memory_op)
            architecture_rank=ConnectionRule.ModifiedConnectionRule(architecture_suggestion,memory_arch)
            
            
            print("selected_operator:",selected_operator)
            print("architecture_rank:",architecture_rank)

            llm_configured_autognn = LLMConfiguredAutoGNN(most_similar, selected_operator,architecture_rank,use_parser=args.use_parser, k=k, mode=mode)
            candidate_pools = llm_configured_autognn.generate_candidate_pools()

            t0 = time.time()
            models_info = candidate_pools[unseen_dataset_name]
            print("candidate_pools:",candidate_pools)
            print("unseen_dataset_name:",unseen_dataset_name)
            print("models_info:",models_info)
            suggested_design_dict_list = []
            for i in range(args.s):
                print("i:",i)
                file_path = os.path.join(response_save_path, f"2_{k}_suggested_design_response_{i}_{current_time}.txt")
                suggested_design_dict,initial_prompt = llm_configured_autognn.suggest_initial_trial(unseen_dataset_name, 
                                                                                     [models_info[i]],selected_operator=selected_operator,architecture_rank=architecture_rank,architecture_suggestion=architecture_suggestion,model_suggestion=model_suggestion,
                                                                                     langchain_query=langchain_query,
                                                                                     similarities=similarity_scores,
                                                                                     file_path=file_path)
                suggested_design_dict_list.append(suggested_design_dict)
                print("suggested_design_dict_list:",suggested_design_dict_list)
            print(f'Suggested design dict: {suggested_design_dict_list}')
            latency3 = time.time() - t0
            print(f"Suggest Initial Trial Latency: {latency3}\n")
        else:
            # Do not transfer knowledge
            t0 = time.time()
            # with open('key.txt', 'r') as file:
            #     langchain_query = ChatOpenAI(api_key=file.read().strip(),
            #                                 temperature=0,
            #                                 model='gpt-3.5-turbo-0125')
            langchain_query = ChatZhipuAI(temperature=0.1,
                                        model='glm-4-plus')
            llm_configured_autognn = LLMConfiguredAutoGNN(None,use_parser=args.use_parser, 
                                                          k=0, add_statistics=args.add_statistics)
            file_path = os.path.join(response_save_path, f"2_simple_suggested_design_response_{current_time}.txt")
            suggested_design_dict = llm_configured_autognn.suggest_initial_trial(unseen_dataset_name, None,
                                                                                 langchain_query=langchain_query,
                                                                                 description=unseen_dataset_desc,
                                                                                 file_path=file_path)
            suggested_design_dict_list = [suggested_design_dict]
            print(f'suggested_design_dict: {suggested_design_dict_list[0]}')
            latency3 = time.time() - t0
            print(f"Suggest Initial Trial Latency: {latency3}\n")
    t0 = time.time()
    initial_detailed_infos = None
    initial_detailed_infos_list = []
    
    if benchmarking:
        print("Benchmarking mode enabled for the initial trial.")
        unseen_dataset_reader = DatasetReader(unseen_dataset_name, benchmark_dir)
        data = unseen_dataset_reader.read_dataset()
        for suggested_design_dict in suggested_design_dict_list:
            initial_detailed_infos = llm_configured_autognn.extract_benchmark_results(unseen_dataset_name, suggested_design_dict, log=True)
            initial_detailed_infos_list.append(initial_detailed_infos)
    else:
        unseen_dataset_reader = DatasetReader(unseen_dataset_name, unseen_dir)

    if initial_detailed_infos is None:
        for suggested_design_dict in suggested_design_dict_list:
            #initial_detailed_infos = llm_configured_autognn.run_gnn_experiment(unseen_dataset_name, data, suggested_design_dict)
            initial_detailed_infos = llm_configured_autognn.run_gnn_experiment(unseen_dataset_name, suggested_design_dict,cuda_number=args.cuda_number)
            initial_detailed_infos_list.append(initial_detailed_infos)
    print("initial_detailed_infos_list:",initial_detailed_infos_list)
        
    best_perf_dict = min(initial_detailed_infos_list, key=lambda x: x['perf'])
    print("best_perf_dict:",best_perf_dict)
    print(f"LLM-suggested Initial Trial:")
    print(f"- Architecture: {best_perf_dict['link']}, Operations: {best_perf_dict['ops']}")
    print(f"- Performance: {best_perf_dict['perf']}\n")
    quit()
    # Step 3.3: Model Proposal Refinement --------------------------------------------------------
    file_path = os.path.join(response_save_path, f"{args.search_strategy}_{current_time}.txt")
    if langchain_query is None:
        # with open('key.txt', 'r') as file:
            # langchain_query = ChatOpenAI(api_key=file.read().strip(),
            #                              temperature=0,
            #                              model='gpt-3.5-turbo-0125')
        langchain_query = ChatZhipuAI(temperature=1,
                                        model='glm-4-plus')
    best_detailed_infos, gnas_history = llm_configured_autognn.run_gnas_pipeline(unseen_dataset_name, 
                                                                                 initial_detailed_infos_list,
                                                                                 max_iter=args.max_iter,
                                                                                 n=args.n,
                                                                                 langchain_query=langchain_query,
                                                                                 search_strategy=args.search_strategy,
                                                                                 file_path=file_path,
                                                                                 selected_operator=selected_operator,
                                                                                 architecture_rank=architecture_rank,
                                                                                 initial_prompt=initial_prompt,
                                                                                 num_children=args.n_children,
                                                                                 cuda_number=args.cuda_number,
                                                                                 benchmarking=benchmarking,
                                                                                 no_reorder=args.no_reorder,
                                                                                 llm_no_candidates=args.llm_no_candidates
                                                                                 )
    print(f"LLM-suggested best model:")
    print(f"- Architecture: {best_detailed_infos['link']}, Operations: {best_detailed_infos['ops']}")
    print(f"- Performance: {best_detailed_infos['perf']}")
    performance_list = [min(entry['perf'] for entry in gnas_history[iter_num]) 
                        if isinstance(gnas_history[iter_num], list) 
                        else gnas_history[iter_num]['perf'] for iter_num in gnas_history]
    best_list = [min(entry['best'] for entry in gnas_history[iter_num]) 
                 if isinstance(gnas_history[iter_num], list) 
                 else gnas_history[iter_num]['best'] for iter_num in gnas_history]
    print(f"DesiGNN History (Real):")
    print(performance_list)
    print(f"DesiGNN History (Best):")
    print(best_list)
    if 'designn' in args.search_strategy and args.benchmarking:
        promoted_list = [gnas_history[iter_num]['promoted']['perf'] 
                         if isinstance(gnas_history[iter_num], dict) and gnas_history[iter_num]['promoted'] is not None 
                         else 0 for iter_num in gnas_history]
        print(f"DesiGNN History (Promoted):")
        print(promoted_list)
    latency4 = time.time() - t0
    print(f"DesiGNN Latency: {latency4}")
    # --------------------------------------------------------------------------------------------

    # Save Result --------------------------------------------------------------------------------
    file_name = f"gnas_result_summary_{current_time}.txt"
    file_path = os.path.join(response_save_path, file_name)
    content = (
        "LLM-suggested best Trial:\n"
        f"- Architecture: {best_detailed_infos['link']}, Operations: {best_detailed_infos['ops']}\n"
        f"- Performance: {best_detailed_infos['perf']}\n"
        "DesiGNN History:\n"
        f"gnas_history: {gnas_history}\n\n"
        f"Graph Dataset Understanding Latency: {latency1}\n"
        f"Graph Dataset Comparison Latency: {latency2}\n"
        f"Suggest Initial Trial Latency: {latency3}\n"
        f"DesiGNN Latency: {latency4}\n"
    )
    with open(file_path, 'w') as file:
        file.write(content)
    
    file_path = os.path.join(response_save_path, f"performances_{current_time}.txt")
    with open(file_path, "w") as file:
        for best in performance_list:
            file.write(f"{best}, ")
        file.write("\n")
        for best in best_list:
            file.write(f"{best}, ")
        file.write("\n")
        if 'designn' in args.search_strategy and args.benchmarking:
            for best in promoted_list:
                file.write(f"{best}, ")
        file.write("\n")

    # Save plot
    file_name = f"performance_vs_iteration_{current_time}.png"
    file_path = os.path.join(response_save_path, file_name)
    if 'designn' in args.search_strategy:
        llm_configured_autognn.plot_performance_vs_iteration(gnas_history, file_path, args.dataset, 
                                                             args.search_strategy)
    else:
        raise ValueError("Unsupported strategy specified.")
    # ------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()

