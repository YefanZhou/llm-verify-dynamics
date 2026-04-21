import numpy as np
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gt-label", type=lambda v: v.lower() in ('yes', 'true', 't', '1'), default=False)
args = parser.parse_args()

gt_label = args.gt_label
run_times = 8

def load_problems_npz(file_path):
    data = np.load(file_path)
    problems = [(str(src), int(idx)) for src, idx in zip(data['dataset_sources'], data['dataset_indices'])]
    return problems

path = "/mnt/ssd2/yefan/llm-verify-dynamics/generator_data/math/model_gen_diff_math.npy"
dataframe = np.load(path, allow_pickle=True).item()
model_performance_dicts = {}
for model_name, model_data in dataframe.items():
    if model_name != 'mean_across_models':
        performance_dict = {}
        for item in model_data:
            key = (item[0], item[1])
            value = item[2]
            performance_dict[key] = value
        model_performance_dicts[model_name] = performance_dict


false_save_path = "cache_correct/false_problems.npz"
true_save_path = "cache_correct/true_problems.npz"
false_loaded = load_problems_npz(false_save_path)
true_loaded = load_problems_npz(true_save_path)
verifier_generator_data = np.load(f'results_correct/verifier_generator_gt{gt_label}.npy', allow_pickle=True).item()
select_load = true_loaded if gt_label else false_loaded


model_averages = {}
for model_name, model_data in dataframe.items():
    if model_name != 'mean_across_models':
        third_values = [model_performance_dicts[model_name][(dataset, dataset_idx)]
                        for dataset, dataset_idx in select_load]
        print('third_values', len(third_values))
        model_averages[model_name] = sum(third_values) / len(third_values)

# Sort models by average generator difficulty (worst to best)
ranked_models = sorted(model_averages.items(), key=lambda x: x[1], reverse=True)
ranked_models = ranked_models[::-1]
print("Models ranked by average performance (worst to best):")
print("=" * 60)
for rank, (model_name, avg_perf) in enumerate(ranked_models, 1):
    print(f"{rank:2d}. {model_name:<45} {avg_perf:.6f}")

print("\n" + "=" * 60)
print(f"Best performing model: {ranked_models[0][0]} ({ranked_models[0][1]:.6f})")
print(f"Worst performing model: {ranked_models[-1][0]} ({ranked_models[-1][1]:.6f})")
print(f"Performance gap: {ranked_models[0][1] - ranked_models[-1][1]:.6f}")

ranked_model_names = [model[0] for model in ranked_models]
print(f"\nRanked model names:")
for i, name in enumerate(ranked_model_names, 1):
    print(f"{i:2d}. {name}")

verifier_model_names = ranked_model_names
true_negative_rate_dict = {}
for verifier_name in verifier_model_names:
    true_negative_rate_dict[verifier_name] = {}
    for generator_name in ranked_model_names:
        true_negative_rate_dict[verifier_name][generator_name] = []


random.seed(42)
np.random.seed(42)
for verifier_name in verifier_model_names:
    for generator_name in ranked_model_names:
        for _ in range(run_times):
            dictionary_lst = verifier_generator_data[verifier_name][generator_name]
            correct = 0
            total = 0
            for item in dictionary_lst:
                idx = random.choice(range(len(item['label'])))
                if gt_label:
                    if item['pred'][idx] == 1:
                        correct += 1
                    elif item['pred'][idx] == -1:
                        correct += 0.5
                else:
                    if item['pred'][idx] == 2:
                        correct += 1
                    elif item['pred'][idx] == -1:
                        correct += 0.5
                total += 1
            true_negative_rate_dict[verifier_name][generator_name].append((correct, total))

np.save(f'results_correct/verifier_generator_gtlabel_{gt_label}_dict.npy', true_negative_rate_dict)
