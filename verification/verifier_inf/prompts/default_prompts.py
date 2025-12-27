PROMPT_VERIFY_SYSTEM = """ 
Please act as an impartial judge and evaluate the correctness of the response provided by an AI assistant to the user prompt displayed below. You will be given the assistant's response. 

When evaluating the assistant's response, identify any mistakes or inaccurate information. Be as objective as possible. Avoid any biases, such as order of responses, length, or stylistic elements like formatting.

Before providing an your final verdict, think through the judging process and output your thoughts as an explanation

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. The response is correct: [[Correct]]
2. The response is incorrect: [[Incorrect]]

Use the following template:
Explanation: Your detailed thought process as an explanation.
Verdict: [[Correct]] or [[Incorrect]].
""".strip()


PROMPT_VERIFY="""
<|User Prompt|>
{question}

<|The Start of Assistant's Answer|>
{response}
<|The End of Assistant's Answer|>
""".strip()


def render_single_instance_verify_prompt(response, question, prompt_strategy='vanilla', judge_model=''):
    if prompt_strategy == 'vanilla':
        sys_prompt = PROMPT_VERIFY_SYSTEM

    prompt_template = PROMPT_VERIFY

    prompt_formatted = prompt_template.format(
        question=question,
        response=response
    )
    
    if 'gemma' in judge_model:
        messages = [{"role": "user", "content": f"{sys_prompt}\n\n{prompt_formatted}"}]
    else:
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt_formatted}]

    return messages



def parse_single_instance_verify(judge_output):

    verdict = judge_output.split('Verdict:')[-1].strip()
    
    correct_indicators = ['[[Correct]]', '[Correct]', 'Correct']
    incorrect_indicators = ['[[Incorrect]]', '[Incorrect]', 'Incorrect']
    has_correct = any(indicator in verdict for indicator in correct_indicators)
    has_incorrect = any(indicator in verdict for indicator in incorrect_indicators)

    if has_correct and not has_incorrect:
        judgement = 'Correct'
    elif has_incorrect and not has_correct:
        judgement = 'Incorrect'
    else:
        judgement = 'Error'

    if judgement == 'Correct':
        return 1
    elif judgement == 'Incorrect':
        return 2
    else:
        return -1

