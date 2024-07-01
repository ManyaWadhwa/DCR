import openai
from transformers import pipeline, LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from other_feedback_model_prompts import make_shepherd_prompt, make_ultracm_prompt
import torch

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
    client = openai.Client()
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


def load_model(model_path):
    """
    If the model name is a generic llaam2/llama3/gpt4 then it is probably a non FT model
    other it is probably a FT model so load it with the path
    :param model_path:
    :return:
    """
    if model_path == "llama2":
        print(f"Loading Llama2-7b-Chat from HF..")
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        generator = pipeline(
            "text-generation", model=model_id, device_map="auto", max_new_tokens=2048
        )
    elif model_path == "llama3":
        print(model_path)
        print(f"Loading Llama3-8b-Instruct from HF..")
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        generator = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            max_new_tokens=2048,
        )
    elif model_path == "ultracm":
        print(f"Loading UltraCM from HF..")
        tokenizer = LlamaTokenizer.from_pretrained("openbmb/UltraCM-13b")
        model = LlamaForCausalLM.from_pretrained("openbmb/UltraCM-13b", device_map="auto")
        generator = pipeline("text-generation", model=model, max_new_tokens=2048, tokenizer=tokenizer)
    elif model_path == "gpt4":
        print(f" GPT-4 Generator..")
        # model_id = "gpt-4-01250-preview"
        model_id = "gpt-4-0613" # TODO Make this a parameter
        generator = get_gpt4_response
    elif model_path == "shepherd":
        print(f"Loading Shepherd from HF..")
        generator = pipeline("text-generation", model="reciprocate/shepherd-13b", max_new_tokens=126,
                             device_map='auto')
    else:
        print(f"Loading {model_path} from HF..")

        generator = pipeline(
            "text-generation", model=model_path, max_new_tokens=2048, device_map="auto"
        )
        print(f"{generator.device=}")
    return generator


def make_final_prompt(model_path, user_message, instruction=None):
    if model_path == "llama2":
        return make_llama2_prompt(system=DEFAULT_SYSTEM_PROMPT, user_message=user_message)
    elif model_path == "llama3":
        return make_llama3_prompt(system=DEFAULT_SYSTEM_PROMPT, user_message=user_message)
    elif model_path == "gpt4":
        return user_message
    elif model_path == "ultracm":
        assert instruction
        return make_ultracm_prompt(instruction, user_message)
    elif model_path == "shepherd":
        assert instruction
        return make_shepherd_prompt(instruction, user_message)
    elif "llama3" in model_path or "L3" in model_path:
        print("Using LLAMA3 system prompt")
        return make_llama3_prompt(system=DEFAULT_SYSTEM_PROMPT, user_message=user_message)
    else:
        print("Using LLAMA2 system prompt")
        return make_llama2_prompt(system=DEFAULT_SYSTEM_PROMPT, user_message=user_message)


def run_inference(generator, prompt, model_name, do_sample=False):
    if model_name == "gpt4":
        return generator(prompt).strip()
    elif model_name == "ultracm":
        response = generator(prompt, num_return_sequences=1, return_full_text=False, handle_long_generation="hole",
                             temperature=1.0, top_p=1.0, max_new_tokens=1024, repetition_penalty=1.2, do_sample=True)
        return response[0]["generated_text"].strip("\n").strip().replace(prompt, "").strip()
    return generator(prompt, do_sample=do_sample)[0]["generated_text"].replace(prompt, "").strip()