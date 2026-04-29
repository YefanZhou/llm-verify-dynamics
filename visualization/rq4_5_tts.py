import os
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

# Make results path a hyperparameter
results_hyper_path = 'results_correct'
cache_correct_path = 'cache_correct'

# Ensure results_hyper_path and cache_correct directories exist
Path(results_hyper_path).mkdir(parents=True, exist_ok=True)
Path(cache_correct_path).mkdir(parents=True, exist_ok=True)

model_lst = [
'Qwen/Qwen2.5-3B-Instruct', 
'Qwen/Qwen2.5-7B-Instruct',
'Qwen/Qwen2.5-72B-Instruct',
'meta-llama/Llama-3.2-3B-Instruct',
'meta-llama/Llama-3.1-8B-Instruct', 
'meta-llama/Llama-3.3-70B-Instruct', 
'google/gemma-2-2b-it', 
'google/gemma-2-9b-it', 
'google/gemma-2-27b-it', 
'mistralai/Mistral-Small-24B-Instruct-2501', 
'mistralai/Ministral-8B-Instruct-2410',  
'Qwen/Qwen3-4B', 
'Qwen/Qwen3-8B', 
'Qwen/Qwen3-32B',
'gpt-4o'
]

bin_model_key = 'mean'
compute_bins = True
verifier = 'gpt-4o'
version = 'v1'

def group_by_score_ranges(data_list):
    ranges = {
        1.0: [],           # exactly 1.0
        '[0.9,1.0)': [],   # >= 0.9 and < 1.0
        '[0.8,0.9)': [],   # >= 0.8 and < 0.9
        '[0.7,0.8)': [],   # >= 0.7 and < 0.8
        '[0.6,0.7)': [],   # >= 0.6 and < 0.7
        '[0.5,0.6)': [],   # >= 0.5 and < 0.6
        '[0.4,0.5)': [],   # >= 0.4 and < 0.5
        '[0.3,0.4)': [],   # >= 0.3 and < 0.4
        '[0.2,0.3)': [],   # >= 0.2 and < 0.3
        '[0.1,0.2)': [],   # >= 0.1 and < 0.2
        '(0.0,0.1)': [],   # > 0.0 and < 0.1
        0.0: []            # exactly 0.0
    }
    
    for item in data_list:
        score = item[2]
        
        if score == 1.0:
            ranges[1.0].append(item)
        elif score >= 0.9:
            ranges['[0.9,1.0)'].append(item)
        elif score >= 0.8:
            ranges['[0.8,0.9)'].append(item)
        elif score >= 0.7:
            ranges['[0.7,0.8)'].append(item)
        elif score >= 0.6:
            ranges['[0.6,0.7)'].append(item)
        elif score >= 0.5:
            ranges['[0.5,0.6)'].append(item)
        elif score >= 0.4:
            ranges['[0.4,0.5)'].append(item)
        elif score >= 0.3:
            ranges['[0.3,0.4)'].append(item)
        elif score >= 0.2:
            ranges['[0.2,0.3)'].append(item)
        elif score >= 0.1:
            ranges['[0.1,0.2)'].append(item)
        elif score > 0.0:  # > 0.0 and < 0.1
            ranges['(0.0,0.1)'].append(item)
        else:  # score == 0.0
            ranges[0.0].append(item)
    
    return ranges

def compute_mean_across_models(model_data_gen_diff, model_lst):
    """
    Compute the mean of the third element (score) across all models 
    for tuples that match on the first two elements (dataset, index).
    """
    
    # Dictionary to collect all scores for each (dataset, index) pair
    tuple_scores = defaultdict(list)
    
    # Collect all tuples from all models
    for model in model_lst:
        if model in model_data_gen_diff:
            for dataset, index, score in model_data_gen_diff[model]:
                key = (dataset, index)  # Use first two elements as key
                tuple_scores[key].append(score)
        else:
            print(f"Warning: Model {model} not found in model_data_gen_diff")
    
    # Get the original order from the first model that exists
    first_model = None
    for model in model_lst:
        if model in model_data_gen_diff:
            first_model = model
            break
    
    if first_model is None:
        print("Error: No models found in model_data_gen_diff")
        return [], {}
    
    # Get original order from first model
    # original_order = [(dataset, index) for dataset, index, _ in model_data_gen_diff[first_model]]
    num_models = len([m for m in model_lst if m in model_data_gen_diff])
    
    # Compute means, preserving original order and alerting for missing data
    mean_results = []
    for dataset, index, _ in model_data_gen_diff[first_model]:  # Use original order from first model
        key = (dataset, index)
        if key in tuple_scores:
            if len(tuple_scores[key]) != num_models:
                print(f"WARNING: Tuple {key} exists in {len(tuple_scores[key])}/{num_models} models")
            mean_score = statistics.mean(tuple_scores[key])
            mean_results.append((dataset, index, mean_score))
        else:
            print(f"ERROR: Tuple {key} not found in any model data")
    
    print(f"Found {len(tuple_scores)} unique tuple keys")
    print(f"Generated {len(mean_results)} mean results from {num_models} models")
    
    return mean_results, tuple_scores


model_data_gen_diff = np.load('/mnt/ssd2/yefan/llm-verify-dynamics/generator_data/math/model_gen_diff_math.npy', allow_pickle=True).item()

model_data_gen_diff_renew = {}
for key in model_lst:
    model_data_gen_diff_renew[key] = model_data_gen_diff[key].copy()

model_data_gen_diff = model_data_gen_diff_renew

model_data_dict_diff = {}
for model_name, data_list in model_data_gen_diff.items():
    # Convert list of tuples (dataset_source, dataset_idx, value) to dictionary
    # Key: (dataset_source, dataset_idx), Value: value
    model_data_dict_diff[model_name] = {(item[0], item[1]): item[2] for item in data_list}

mean_results, all_tuple_scores = compute_mean_across_models(model_data_gen_diff, model_lst)

# Add to your model_data_gen_diff dictionary
model_data_gen_diff["mean"] = mean_results
model_data_gen_diff["mean_model_lst"] = model_lst

if compute_bins:
    grouped_data_all = {}

    print(f"\nProcessing {bin_model_key}...")
    
    # Sort data for this generator
    sorted_list_desc = sorted(model_data_gen_diff[bin_model_key], key=lambda x: x[2], reverse=True)
    
    # Group by score ranges
    grouped_data = group_by_score_ranges(sorted_list_desc)
    
    # Store the grouped data
    grouped_data_all[bin_model_key] = grouped_data
    
    # Print summary for this generator
    print(f"Summary for {bin_model_key}:")
    total = 0
    for range_key, items in grouped_data.items():
        if items:
            print(f"  {range_key}: {len(items)} items")
            total += len(items)
    print(f"  Total: {total} items")

    save_path = f"{cache_correct_path}/model_data_gen_diff_grouped_{bin_model_key.replace('/', '_')}_{version}_subsample_rejectall_dice.npy"
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    np.save(save_path, grouped_data_all)

print('grouped_data_all', grouped_data_all.keys())


first_key = list(grouped_data_all.keys())[0] #grouped_data_all


bin_generator_diff_lst = {}
for bin_key in grouped_data_all[first_key]:
    bin_generator_diff_lst[bin_key] = {}
    for generator_name in model_lst:
        bin_generator_diff_lst[bin_key][generator_name] = []

for generator_key in model_lst:
    generator_file = generator_key.replace('/', '_').replace('-', '_')
    path = "/mnt/ssd2/yefan/llm-verify-dynamics/verifier_data/" + \
    f"{verifier}/single_verification/temp0.0_topp1.0_seqs1/vanilla/" + \
    f"verification_sample64_{generator_file}/eval_detailed_verification_sample64_{generator_file}.jsonl"

    if not os.path.exists(path):
        print(f"{path} doesn't exist!!!!!")
        raise NotImplementedError

    df = pd.read_json(path, lines=True)
    
    verifier_gene_score = {}
    
    for bin_key in grouped_data_all[bin_model_key]:
        if bin_key == 1.0 or bin_key == 0.0:
            continue

        score_lst = []
        subgroup = grouped_data_all[bin_model_key][bin_key]
        
        
        for dataset_source, dataset_idx, diff_v in tqdm.tqdm(subgroup, total=len(subgroup)):
            df_of_the_prob = df[(df['dataset_source'] == dataset_source) & (df['dataset_idx'] == dataset_idx)]

            y_pred_raw = df_of_the_prob['votes'].apply(lambda x: x[0]).values
            y_pred = (y_pred_raw == 1).astype(int)
            y_true = df_of_the_prob['label'].astype(int).values
            
            tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
            tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives  
            fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            assert len(y_pred_raw) == len(y_true)
            true_pos_pred_half = 0
            true_neg_pred_half = 0
            for y_pred_vote, y_true_label in zip(y_pred_raw, y_true):
                if y_pred_vote == 1 and y_true_label == 1:
                    true_pos_pred_half += 1
                elif y_pred_vote == 2 and y_true_label == 0:
                    true_neg_pred_half += 1
                elif y_pred_vote == -1:
                    if y_true_label == 1:
                        true_pos_pred_half += 0.5
                    elif y_true_label == 0:
                        true_neg_pred_half += 0.5

            true_count = np.sum(y_true == 1)
            false_count = np.sum(y_true == 0)

            tpr = tp / true_count if true_count > 0 else np.nan
            tnr = tn / false_count if false_count > 0 else np.nan

            local_votes = [item[0] for item in df_of_the_prob['votes'].tolist()]
            # bool to int True=1 False = 0
            local_labels = [int(item) for item in df_of_the_prob['label'].tolist()]
            
            denominator = 0
            nominator = 0
            for local_v, local_l in zip(local_votes, local_labels):
                if local_v == 1:
                    denominator += 1
                    if local_l == 1:
                        nominator += 1
                    elif local_l == 0:
                        nominator += 0
                elif local_v == -1:
                    denominator += 0.5
                    if local_l == 1:
                        nominator += 0.5
                    elif local_l == 0:
                        nominator += 0

            original_labels_array = df_of_the_prob['label'].astype(int).values
            
            average_original = original_labels_array.mean()
            maximum_original = original_labels_array.max()
            minimum_original = original_labels_array.min() 
            
            if denominator == 0:
                average = average_original
            else:
                average = nominator / denominator

            assert abs(average_original - model_data_dict_diff[generator_key][(dataset_source, dataset_idx)]) < 1e-5, (generator_key, dataset_source, dataset_idx)
            bin_generator_diff_lst[bin_key][generator_key].append(model_data_dict_diff[generator_key][(dataset_source, dataset_idx)])

            score_lst.append({'average': average,
                            'average_original':average_original,
                            'maximum_original':maximum_original,
                            'minimum_original':minimum_original,
                            'dataset_source': dataset_source,
                            'dataset_idx': dataset_idx,
                            'diff_v': diff_v, 
                            'tpr':  tpr, 
                            'tnr':  tnr, 
                            'tp':   tp, 
                            'tn':   tn, 
                            'fp':   fp, 
                            'fn':   fn,
                            'y_true': y_true,
                            'y_pred': y_pred,
                            'true_pos_pred_half': true_pos_pred_half,
                            'true_neg_pred_half': true_neg_pred_half,
                            'true_count': true_count,
                            'false_count': false_count
                            })
        
        verifier_gene_score[bin_key] = score_lst
        
        path = f"{results_hyper_path}/subsample_rejectall_dice_verifier_{verifier}_w_dataindex_label_meandiff_bins_{bin_model_key.replace('/', '_')}_{version}"
        Path(path).mkdir(parents=True, exist_ok=True)
        np.save(f"{path}/bin_generator_diff_lst_binmodel_{bin_model_key.replace('/', '_')}.npy", bin_generator_diff_lst)
        

    path = f"{results_hyper_path}/subsample_rejectall_dice_verifier_{verifier}_w_dataindex_label_meandiff_bins_{bin_model_key.replace('/', '_')}_{version}"
    Path(path).mkdir(parents=True, exist_ok=True)
    np.save(f'{path}/gene_score_{generator_file}.npy', verifier_gene_score)
    np.save(f"{path}/bin_generator_diff_lst_binmodel_{bin_model_key.replace('/', '_')}.npy", bin_generator_diff_lst)