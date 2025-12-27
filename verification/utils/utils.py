import json,os,random,torch,time
import numpy as np
import random
from openai import OpenAI
from collections import defaultdict



# API setting constants
API_MAX_RETRY = 25
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

def chat_completion_openai(model, messages, temperature, max_tokens):
    client = OpenAI()
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            # remove system prompt for o1 models, use allowable sampling parameters
            if "o1" in model or "o3-" in model:
                if messages[0]['role'] == 'system':
                    messages = messages[1:]
                response = client.chat.completions.create(
                    model=model, messages=messages, n=1, temperature=1
                )

            else:
                response = client.chat.completions.create(
                    model=model, messages=messages, n=1, temperature=temperature, max_tokens=max_tokens
                )

            output = response.choices[0].message.content
            break
        except openai.APIError as e:
            # Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            time.sleep(API_RETRY_SLEEP)

        except openai.APIConnectionError as e:
            # Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            time.sleep(API_RETRY_SLEEP)

        except openai.RateLimitError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            time.sleep(API_RETRY_SLEEP)

    return output



def read_jsonl(file_path):
    with open(file_path, 'r') as fr:
        data = [json.loads(line) for line in fr.readlines()]
    print(len(data))
    print(data[0].keys())
    return data

def write_jsonl(data, jfile, skip_none=False):
    with open(jfile, 'w', encoding='utf-8') as f:
        for d in data:
            if d is None and not skip_none:
                raise ValueError('None object !')
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f'Wrote {len(data)} -> {jfile}')
    
def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def check_results_exist(args, save_dir):
    # if dataset is not reasoningjudgebench/1 split, check if fp exists
    if args.dataset_name != 'reasoningjudgebench':
        output_path = os.path.join(save_dir, f'eval_detailed_{args.dataset_name}.jsonl')        
        return os.path.exists(output_path), [output_path]

    # else, check if ALL splits of reasoningjudgebench exist
    else:
        split_names = [
            'aime_4o_pairwise',
            'aime_pairwise',
            'arc_challenge_pairwise',
            'bbeh_be_pairwise',
            'bbeh_cu_pairwise',
            'bbeh_dq_pairwise',
            'bbeh_hyper_pairwise',
            'bbeh_ma_pairwise',
            'reclor_pairwise',
            'strategy_qa_pairwise',
            'folio_pairwise',
            'supergpqa_pairwise',
            'olympiadbench_pairwise',
        ]
        output_paths = [os.path.join(save_dir, f'eval_detailed_{split_name}.jsonl') for split_name in split_names]
        return all([os.path.exists(op) for op in output_paths]), output_paths


# Takes a list of votes [1, 2] WITHOUT INVALID VOTES, returns majority votes with arbitrary tie break
def get_majority_vote(votes, weights = []):
    if votes == []:
        return -1

    is_weighted = True
    if weights == []:
        weights = [1 for _ in range(len(votes))]
        is_weighted = False

    ctr = defaultdict(lambda: 0)
    for v, w in zip(votes, weights):
        ctr[v] += w

    max_val = max(ctr.values())
    max_keys = [k for k, v in ctr.items() if v == max_val]

    # If unweighed and no clear majority, return invalid
    # e.g., 1 has 4 votes, 2 has 4 votes --> tie --> invalid
    if not is_weighted and max_val <= len(votes) / 2:
        return -1

    return random.choice(max_keys)


            
def compute_acc(judgements_1, judgements_2, labels, metadata = []):
    # Compute accuracy metrics
    accuracy_consistent = []
    accuracy_swap1 = []
    accuracy_swap2 = []
    consistency = []
    for idx, (j1, j2, label) in enumerate(zip(judgements_1, judgements_2, labels)):

        assert isinstance(j1, list) and isinstance(j2, list)
    
        # Get weights if they exist
        if metadata != []:
            weights = metadata[idx].get('weights', [])
            if weights != [] and len(weights) == len(j1):
                weights = [[w[0] for w in weights], [w[1] for w in weights]]
            else:
                weights = [[], []]
        else:
            weights = [[], []]

        # Get majority vote over consistency runs
        j1_maj = get_majority_vote(j1, weights=weights[0])
        j2_maj = get_majority_vote(j2, weights=weights[1])

        # Compute accuracy
        if j1_maj == j2_maj:
            accuracy_consistent.append([int(j1_maj == label)])
            consistency.append([1])
        # if one response is tie and one response is clear, take the clear 
        # Tie = 0
        elif (j1_maj == 0 and j2_maj != 0) or (j1_maj != 0 and j2_maj == 0):
            for jment in [j1_maj, j2_maj]:
                if jment != 0:
                    accuracy_consistent.append([int(jment == label)])
                    consistency.append([1])
                    break
            
        else:
            accuracy_consistent.append([0])
            consistency.append([0])

        accuracy_swap1.append([int(j1_maj == label)])
        accuracy_swap2.append([int(j2_maj == label)])
        


    return accuracy_consistent, consistency, accuracy_swap1, accuracy_swap2


