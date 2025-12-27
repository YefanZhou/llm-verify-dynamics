import importlib




def load_prompter(model):
    print(f"Model: {model}")
    
    model_short = 'default_prompts'
    print(f"!!!!Loading prompter from {model_short}.py!!!!!")

    prompter = importlib.import_module('verifier_inf.prompts.{}'.format(model_short))
    return prompter


