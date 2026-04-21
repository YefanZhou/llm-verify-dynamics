import numpy as np
import time
import pandas as pd
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Evaluate model on a dataset')
parser.add_argument('--problem-select', type=str2bool, default=False)
parser.add_argument('--ans-select', type=str2bool, default=False)
parser.add_argument('--gt-label', type=str2bool, default=False)
args = parser.parse_args()


judge_path_3b = "/mnt/ssd2/yefan/llm-verify-dynamics/verifier_data/Qwen/Qwen2.5-3B-Instruct/single_verification/temp0.0_topp1.0_seqs1/vanilla/math_balanced_subsample_4.parquet"
gen_diff_path = '/mnt/ssd2/yefan/llm-verify-dynamics/generator_data/math/model_gen_diff_math.npy'
false_save_path = "cache_correct/false_problems.npz"
true_save_path = "cache_correct/true_problems.npz"


def save_problems_npz(problem_list, file_path):
    if not problem_list:
        structured_data = {
            'dataset_sources': np.array([], dtype='U50'),
            'dataset_indices': np.array([], dtype=np.int32),
            'count': 0
        }
    else:
        structured_data = {
            'dataset_sources': np.array([str(item[0]) for item in problem_list], dtype='U50'),
            'dataset_indices': np.array([int(item[1]) for item in problem_list], dtype=np.int32),
            'count': len(problem_list)
        }
    np.savez_compressed(file_path, **structured_data)
    print(f"Saved {structured_data['count']} problems to {file_path}")
    return structured_data

def load_problems_npz(file_path):
    data = np.load(file_path)
    problems = [(str(src), int(idx)) for src, idx in zip(data['dataset_sources'], data['dataset_indices'])]
    return problems


gene_model_name_lst = [
    'Qwen/Qwen2.5-3B-Instruct',
    'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen2.5-72B-Instruct-Turbo',
    'Qwen/Qwen3-4B',
    'Qwen/Qwen3-8B',
    'Qwen/Qwen3-32B',
    'google/gemma-2-2b-it',
    'google/gemma-2-9b-it',
    'google/gemma-2-27b-it',
    'gpt-4o',
    'meta-llama/Llama-3.2-3B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct',
    'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    'mistralai/Ministral-8B-Instruct-2410',
    'mistralai/Mistral-Small-24B-Instruct-2501',
]


if args.problem_select:
    dataframe = np.load(gen_diff_path, allow_pickle=True).item()
    judge_frame = pd.read_parquet(judge_path_3b)
    model_name_lst = judge_frame['model'].unique().tolist()

    start_time = time.time()
    total_items = len(dataframe['mean_across_models'])

    false_problem_lst = []
    true_problem_lst = []
    for i, (dataset_source, dataset_idx, _) in enumerate(dataframe['mean_across_models']):
        matching_rows = judge_frame[
            (judge_frame['dataset_source'] == dataset_source) &
            (judge_frame['dataset_idx'] == dataset_idx)
        ]

        all_models_have_false = True
        all_models_have_true = True
        for model_name in model_name_lst:
            model_matching_rows = matching_rows[matching_rows['model'] == model_name]
            assert len(model_matching_rows) != 0
            labels = model_matching_rows['label'].tolist()

            if not any(label == False for label in labels):
                all_models_have_false = False
            if not any(label == True for label in labels):
                all_models_have_true = False

        if all_models_have_false:
            false_problem_lst.append((dataset_source, dataset_idx))
        if all_models_have_true:
            true_problem_lst.append((dataset_source, dataset_idx))

        if (i + 1) % 200 == 0 or (i + 1) == total_items:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total_items - i - 1) / rate if i + 1 < total_items else 0
            print(f"Progress: {i+1}/{total_items} ({(i+1)/total_items*100:.1f}%) | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s")

    print(f'False problems found: {len(false_problem_lst)}')
    print(f'True problems found: {len(true_problem_lst)}')
    save_problems_npz(false_problem_lst, false_save_path)
    save_problems_npz(true_problem_lst, true_save_path)


if args.ans_select:
    verifier_model_name_lst = gene_model_name_lst
    judge_hyper = 'temp0.0_topp1.0_seqs1'

    result_dict = {v: {g: [] for g in gene_model_name_lst} for v in verifier_model_name_lst}

    false_loaded = load_problems_npz(false_save_path)
    true_loaded = load_problems_npz(true_save_path)
    select_load = true_loaded if args.gt_label else false_loaded

    print(f"False problems loaded: {len(false_loaded)} items")
    print(f"True problems loaded: {len(true_loaded)} items")

    for verifier_name in verifier_model_name_lst:
        verifier_path = (
            f"/mnt/ssd2/yefan/llm-verify-dynamics/verifier_data/{verifier_name}"
            f"/single_verification/{judge_hyper}/vanilla/math_balanced_subsample_4.parquet"
        )
        judge_frame = pd.read_parquet(verifier_path)
        print('read', verifier_path)

        start_time = time.time()
        total_items = len(select_load)

        for i, (dataset_source, dataset_idx) in enumerate(select_load):
            matching_rows = judge_frame[
                (judge_frame['dataset_source'] == dataset_source) &
                (judge_frame['dataset_idx'] == dataset_idx)
            ]
            for generator_name in gene_model_name_lst:
                model_matching_rows = matching_rows[matching_rows['model'] == generator_name]
                preds = np.array([item[0] for item in model_matching_rows['votes'].tolist()])
                labels = np.array(model_matching_rows['label'].tolist())
                indices = np.where(labels == args.gt_label)[0]

                result_dict[verifier_name][generator_name].append({
                    'pred': preds[indices],
                    'label': labels[indices],
                    'dataset_source': dataset_source,
                    'dataset_idx': dataset_idx,
                })

            if (i + 1) % 200 == 0 or (i + 1) == total_items:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (total_items - i - 1) / rate if i + 1 < total_items else 0
                print(f"Progress: {i+1}/{total_items} ({(i+1)/total_items*100:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s")

        for generator_name in gene_model_name_lst:
            print(f'generator_name: {generator_name}', len(result_dict[verifier_name][generator_name]))

        np.save(f'results_correct/verifier_generator_gt{args.gt_label}.npy', result_dict)

# result_dict[verifier][generator]: list of {'pred': [...], 'label': [...], 'dataset_source': ..., 'dataset_idx': ...}
# gt_label=True  -> true positive subset  (label=True,  correct pred=1)
# gt_label=False -> true negative subset  (label=False, correct pred=2)
