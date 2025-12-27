import json,os,argparse
from verifier_inf.verifier import VllmEndpointJudge, APIJudge
from datasets import Dataset, DatasetDict
import datasets

HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)

datasets.disable_caching()

def main(args):

    judge = []
    print(f"Judge model name: {args.judge_model}")
    #https://api.together.xyz/v1/
    #https://api.openai.com/v1/
    
    max_tokens=1024

    if args.provider == 'local':
        url_format='http://localhost:{port}/v1'
        ports = args.ports.split(',')
        print('ports', ports)
        for p in ports:
            if 'localhost' in args.base_url:
                url = url_format.format(port=p)
            else:
                url = args.base_url

            endpoint_config = {
                "base_url": url,
                "api_key": args.api_key,
                "model_name": args.judge_model,
            }
            judge_comp = VllmEndpointJudge(endpoint_config)
            judge.append(judge_comp)
    else:
        api_config = {
                'model': args.judge_model,
                'provider': args.provider,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'max_tokens': max_tokens,
                'num_sequences': args.num_sequences
        }
        
        judge.append(APIJudge(api_config, prompt_strategy=args.prompt_strategy))
        
    
    print('Number of endpoints:', len(judge))


    if args.save_dir is not None:
    # Use the explicitly provided save_dir (includes sampling parameters)
        save_dir = args.save_dir
        print(f"Using provided save_dir: {save_dir}")
    else:
        # Use the default path construction
        save_dir = os.path.join(args.output_dir, args.judge_model, args.prompt_strategy, args.dataset_name)
        print(f"Using default save_dir: {save_dir}")

    if not os.path.isdir(save_dir):
        print(f"Creating directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

    # Inference!
    #results_exist, output_paths = check_results_exist(args, save_dir)
    #if not results_exist or args.force_rerun:
        ####################################
        # Run judge model locally with VLLM!
        ####################################

    
    
    from verifier_inf.scaling_method.standard import StandardJudging
    judging_task = StandardJudging(args.dataset_name, judge, max_tokens, 
                                    debug=args.debug, 
                                    num_proc=args.num_proc, 
                                    n_samples=args.num_sequences, temp=args.temperature, top_p=args.top_p
                                    )

    #-----------------
    # Run task!
    updated_dataset = judging_task.run(args.evaluation_protocol, args.prompt_strategy, provider=args.provider)



    if isinstance(updated_dataset, Dataset):
        updated_dataset = DatasetDict({args.dataset_name: updated_dataset})

    for split_name, updated_ds in updated_dataset.items():
        output_path = os.path.join(save_dir, f'eval_detailed_{split_name}.jsonl')
        output_result_path = os.path.join(save_dir, f'eval_result_{split_name}.json')
        if args.debug:
            output_path = output_path.replace('.jsonl', '.debug.jsonl')
            output_result_path = output_result_path.replace('.json', '.debug.json')

        
        judgements = updated_ds['judgement']
        outputs = updated_ds['output']
        
        with open(output_path, 'w', buffering=1) as fw:
            for i in range(len(updated_ds)):
                
                metadata = updated_ds[i].get('metadata', {})

                output = {
                    "unique_id": updated_ds[i]['id'],
                    "model": updated_ds[i]['model'],
                    "dataset_source":updated_ds[i]['dataset_source'],
                    "dataset_idx":updated_ds[i]['dataset_idx'],
                    "response_idx": updated_ds[i]['response_idx'],
                    "temperature": updated_ds[i]['temperature'],
                    "top_p": updated_ds[i]['top_p'],
                    "sample_num": updated_ds[i]['sample_num'],
                    "max_tokens": updated_ds[i]['max_tokens'],
                    "votes": judgements[i],
                    "label": updated_ds[i]['label'],
                    "question": updated_ds[i]['prompt'],
                    "swap_inference1": outputs[i],
                    "metadata": metadata,
                    "gold_answer": updated_ds[i]['gold_answer'],
                    "response": updated_ds[i]['response']
                }

                if "sampling_strategy" in updated_ds[i]:
                    output["sampling_strategy"] = updated_ds[i]["sampling_strategy"]
            
                fw.write(json.dumps(output)+"\n")

        

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Evaluation parameters
    parser.add_argument("--output_dir", type=str, default='results/', help="the directory for storing results")
    parser.add_argument("--save_dir", type=str, default=None, help="Override save directory path (optional)")
    parser.add_argument("--dataset_name", type=str, default="reasoningjudgebench", help='verification_single_greedy')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--force_rerun", action='store_true')
    parser.add_argument("--num_proc", type=int, default=10, help="number of threads to use for parallel processing of examples for openai api calls")

    # judge info
    parser.add_argument("--judge_model", type=str, help="model checkpoint or name")
    parser.add_argument("--prompt_strategy", type=str, default='vanilla', choices=['vanilla', 'varying-prompt', 'solve-then-verify-anyerror'], help='type of prompting for judge models. Used to enable prompt ablations, if needed')
    parser.add_argument("--evaluation_protocol", type=str, default="single_verification", help="How to conduct evaluation")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="base url for served VLLM judge")
    parser.add_argument("--ports", type=str, help="comma-separated list of ports for different vllm servers")
    parser.add_argument("--api_key", type=str, default="")

    # decoding strategy
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--num_sequences", default=1, type=int)
    parser.add_argument("--provider", default='local', type=str)
    parser.add_argument("--max_tokens", default=1024, type=int, help="Only used for budget forcing task; otherwise set manually")
    parser.add_argument("--max_rounds", default=2, type=int, help="Only used for self reflection (number of reflection rounds)")

    args = parser.parse_args()

    main(args)