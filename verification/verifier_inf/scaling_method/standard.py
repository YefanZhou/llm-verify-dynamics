from vllm import SamplingParams
from verifier_inf.scaling_method import Task

class StandardJudging(Task):
    def __init__(self, dataset_name: str, judge: list, max_tokens: int = 1024, 
                                debug: bool = False, 
                                num_proc: int = 10, 
                                n_samples: int = 1, 
                                temp: float = 0.0, 
                                top_p: float = 1.0):

        self.sampling_params = SamplingParams(
            n=n_samples,
            temperature=temp,
            top_p=top_p,
            max_tokens=max_tokens
        )

        super().__init__(dataset_name, judge, debug=debug, num_proc=num_proc)

    def get_single_verification_judgments(self, model_i: int, response: str, 
                                          question: str, 
                                          prompt_strategy: str,
                                          judge_model: str):

        outputs, _, messages = self.judge[model_i].single_instance_verify(response, 
                                                                question, 
                                                                self.sampling_params, 
                                                                prompt_strategy=prompt_strategy,
                                                                judge_model=judge_model)
        
        judgments = [self.judge[model_i].prompter.parse_single_instance_verify(o) for o in outputs]

        return outputs, judgments, messages
