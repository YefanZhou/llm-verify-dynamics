import numpy as np
from vllm import SamplingParams
from openai import OpenAI
from datasets import Dataset, DatasetDict, load_dataset
from verifier_inf.verifier import Judge
import os
import tqdm
import asyncio

BASE_PATH = os.environ.get('BASE_PATH', '')

# Define the models for verification_sample64 datasets
verification_models = [
    'Qwen_Qwen2.5_3B_Instruct',
    'Qwen_Qwen2.5_7B_Instruct',
    'Qwen_Qwen2.5_72B_Instruct',
    'Qwen_Qwen2.5_72B_Instruct_Turbo',
    'Qwen_Qwen3_4B',
    'Qwen_Qwen3_8B',
    'Qwen_Qwen3_32B',
    'google_gemma_2_2b_it',
    'google_gemma_2_9b_it',
    'google_gemma_2_27b_it',
    'gpt_4o',
    'meta_llama_Llama_3.2_3B_Instruct',
    'meta_llama_Llama_3.1_8B_Instruct',
    'meta_llama_Llama_3.3_70B_Instruct',
    'meta_llama_Llama_3.3_70B_Instruct_Turbo', 
    'mistralai_Ministral_8B_Instruct_2410',
    'mistralai_Mistral_Small_24B_Instruct_2501',
]

dataset_to_path = {
}

# Add verification_sample64 entries using a loop
for model in verification_models:
    key = f'{model}_math_subsample'
    path = f'{BASE_PATH}/generator_data/math/balanced_subsample/{model}.jsonl'
    dataset_to_path[key] = path
    
    key = f'{model}_knowledge_subsample'
    path = f'{BASE_PATH}/generator_data/knowledge/balanced_subsample/{model}.jsonl'
    dataset_to_path[key] = path
    
    key = f'{model}_reasoning_subsample'
    path = f'{BASE_PATH}/generator_data/nl_reasoning/balanced_subsample/{model}.jsonl'
    dataset_to_path[key] = path
    

input_key_maps = {
}

question_key_maps = {
}

for model in verification_models:
    key = f'{model}_math_subsample'
    input_key_maps[key] = ['response']
    question_key_maps[key] = 'prompt'
    
    key = f'{model}_knowledge_subsample'
    input_key_maps[key] = ['response']
    question_key_maps[key] = 'prompt'
    
    key = f'{model}_reasoning_subsample'
    input_key_maps[key] = ['response']
    question_key_maps[key] = 'prompt'


def get_dataset(dataset_name: str, num_proc: int):    
    data_path = dataset_to_path[dataset_name]
    
    if '.jsonl' in data_path:
        eval_dataset = load_dataset("json", data_files=data_path, split='train')
    elif '.parquet' in data_path:
        eval_dataset = load_dataset("parquet", data_files=data_path, split='train')
    else:
        eval_dataset = load_dataset(data_path)
   
    return eval_dataset, question_key_maps[dataset_name], input_key_maps[dataset_name]


class Task():
    def __init__(self, dataset_name: str, judge: list, debug: bool = False, num_proc: int = 10):
        self.judge = judge
        self.num_proc = num_proc
        self.dataset_name=dataset_name
        dataset, question_key, response_keys = get_dataset(dataset_name, num_proc)

        if debug and isinstance(dataset, Dataset):
            dataset = dataset.select(range(10)) 
        # For debugging, we'll just grab the first split
        elif debug and isinstance(dataset, DatasetDict):
            dataset = dataset[list(dataset.keys())[0]]


        self.debug = debug
        self.question_key = question_key
        self.response_keys = response_keys
        self.dataset = dataset  

    
    def run_single_verification(self, prompt_strategy, judge_model):
        def proc_example(example, idx, question_key, response_keys):
            response = example[response_keys[0]]
            question = example[question_key]
            model_idx = idx % len(self.judge)

            outputs, judgments, messages = self.get_single_verification_judgments(model_idx, 
                                                            response, 
                                                            question, 
                                                            prompt_strategy=prompt_strategy,
                                                            judge_model=judge_model)

            if idx == 0:
                print(messages)

            output = {
                'judgement': judgments,
                'output': outputs,
            }

            return output

        updated_dataset = self.dataset.map(
            proc_example, 
            num_proc = self.num_proc, 
            with_indices=True,
            batched=False,
            fn_kwargs={
                "question_key": self.question_key, 
                "response_keys": self.response_keys})

        return updated_dataset


    def run_single_verification_api(self, prompt_strategy, judge_model):
        
        print('enter run_single_verification_api.....')
        self.judge[0].endpoint.show_sampling_params()

        original_messages = []
        for i in tqdm.tqdm(range(len(self.dataset)), desc='rendering messages'):
            example = self.dataset[i]
            response = example[self.response_keys[0]]
            question = example[self.question_key]
            message = self.judge[0].prompter.render_single_instance_verify_prompt(response, question, 
                                                                                  prompt_strategy=prompt_strategy, 
                                                                                  judge_model=judge_model)
            original_messages.append(message)
            

        num_samples = self.judge[0].num_sequences

        if num_samples > 1:
            duplicated_messages = []
            for msg in original_messages:
                duplicated_messages.extend([msg] * num_samples)
            
            print(f"Original messages: {len(original_messages)}")
            print(f"Duplicated messages: {len(duplicated_messages)} (each duplicated {num_samples} times)")
        else:
            duplicated_messages = original_messages

        flat_outputs = asyncio.run(self.judge[0].endpoint.generate(duplicated_messages))
        
        # Group responses back into lists
        grouped_outputs = []
        grouped_judgments = []
        
        if num_samples > 1:
            flat_outputs = [item[0] for item in flat_outputs]
            for i in range(len(original_messages)):
                start_idx = i * num_samples
                end_idx = start_idx + num_samples
                
                # Group outputs
                output_group = flat_outputs[start_idx:end_idx]
                grouped_outputs.append(output_group)
                
                # Parse judgments for this group
                judgment_group = []
                for output in output_group:
                    judgment_group.append(self.judge[0].prompter.parse_single_instance_verify(output))
                grouped_judgments.append(judgment_group)
        else:
            grouped_outputs = flat_outputs
            for o_set in grouped_outputs:
                judge_set = []
                for o in o_set:
                    judge_set.append(self.judge[0].prompter.parse_single_instance_verify(o))
                grouped_judgments.append(judge_set)

        updated_dataset = self.dataset.add_column('messages', original_messages)
        updated_dataset = updated_dataset.add_column('output', grouped_outputs)
        updated_dataset = updated_dataset.add_column('judgement', grouped_judgments)

        return updated_dataset
    

    def run(self, evaluation_protocol, prompt_strategy: str = 'vanilla', provider: str = 'local', judge_model: str = ''):
        
        if evaluation_protocol == 'single_verification' and provider == 'local':
            return self.run_single_verification(prompt_strategy, judge_model)
        elif evaluation_protocol == 'single_verification' and provider != 'local':
            return self.run_single_verification_api(prompt_strategy, judge_model)
        else:
            raise NotImplementedError(f"Protocol [{evaluation_protocol}] not implemented yet!")
    
