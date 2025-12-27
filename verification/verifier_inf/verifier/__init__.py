from vllm import SamplingParams
from openai import OpenAI
from datasets import Dataset
from verifier_inf.verifier.vllm_utils import VllmEndpoint
from verifier_inf.verifier.model_toge import TogetherAIInference
from verifier_inf.prompts import load_prompter


class Judge():
    def __init__(self, prompter):
        self.prompter = prompter

    def generate(self, messages: list):
        raise NotImplementedError


    def single_instance_verify(self, response: str, 
                                question: str, 
                               sampling_params: SamplingParams, 
                               prompt_strategy:str = 'vanilla',
                               return_comp_tokens: bool = False, 
                               completion_gen: bool = False,
                               judge_model: str = ''):
        
        messages = self.prompter.render_single_instance_verify_prompt(response, question, prompt_strategy=prompt_strategy, judge_model=judge_model)
        
        response_texts, comp_tokens = self.generate(messages, sampling_params, return_comp_tokens=return_comp_tokens, completion_gen=completion_gen)

        return response_texts, comp_tokens, messages



    
class VllmEndpointJudge(Judge):
    def __init__(self, endpoint_config: dict, prompter=None, prompt_strategy='vanilla'):
        self.endpoint = VllmEndpoint(endpoint_config)
        self.model_name = self.endpoint.model_name

        if prompter is None:
            prompter = load_prompter(self.model_name)

        print(f"Verify Prompt: {prompter.PROMPT_VERIFY}")

        super().__init__(prompter)

    def generate(self, messages: list, sampling_params: SamplingParams, return_comp_tokens: bool = False, completion_gen: bool = False):
        return self.endpoint.generate(messages, sampling_params, return_comp_tokens = return_comp_tokens, completion_gen = completion_gen)


class APIJudge(Judge):
    def __init__(self, endpoint_config: dict, prompter=None, prompt_strategy='vanilla'):
        self.endpoint = TogetherAIInference(model=endpoint_config['model'], 
                            provider=endpoint_config['provider'],
                            temperature=endpoint_config['temperature'],
                            top_p=endpoint_config['top_p'],
                            max_tokens=endpoint_config['max_tokens'],
                            n=1)
        
        self.model_name = endpoint_config['model']
        self.num_sequences = endpoint_config.get('num_sequences', 1)
        
        if prompter is None:
            prompter = load_prompter(self.model_name)
        
        print(f"Verify Prompt: {prompter.PROMPT_VERIFY}")

        super().__init__(prompter)


