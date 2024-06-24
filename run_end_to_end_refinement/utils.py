import openai

client = openai.Client()

DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."

def make_llama3_prompt(system, user_message):
    prompt = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    return prompt

def get_gpt4_response(user_prompt: str, model="gpt-4-0125-preview"):
    response = client.chat.completions.create(
                model="gpt-4-0125-preview",messages=[ {"role": "user", "content": user_prompt}])
    content = response.choices[0].message.content
    return content


def make_llama2_prompt(
        system: str,
        user_message: str,
) -> str:
    return """<s>[INST] <<SYS>> {system} <</SYS>>
{instruction} [/INST]""".format(system=system, instruction=user_message)