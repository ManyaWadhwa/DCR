"""
This script generates initial completions from llama2-7b-chat
"""
import openai
import os
import torch
import json
import time
from transformers import pipeline
import argparse
from nltk.tokenize import sent_tokenize
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."

tofueval_prompt = """Document: {Document}

Summarize the provided document focusing on "{topic}". The summary should be less than 50 words in length."""

def get_tofueval_response(generator, example, model="llama2"):
    torch.inference_mode()
    topic = example['topic']
    document = example['source_doc']
    user_instruction = tofueval_prompt.format(**{"Document":document,"topic":topic})
    if model=="llama2":
        system = f"<s>[INST]<<SYS>>{DEFAULT_SYSTEM_PROMPT}<</SYS>>\n\n"
        prompt = system + user_instruction + "[/INST]"
    elif model == "llama3":
        prompt = get_llama3_prompt(system=DEFAULT_SYSTEM_PROMPT, user_message=user_instruction)

    attempts = 3
    current_attempt = 0
    valid = False
    # print(prompt)
    while current_attempt<attempts and not valid:
        # resample till we get a valid response
        current_attempt += 1
        response = generator(prompt)
        response = response[0]["generated_text"].replace(prompt, "").strip()
        sentences = sent_tokenize(response)
        filter_out = "Љ"
        if filter_out in response or len(sentences) > 20:
            valid = False
        else:
            valid = True
        print("not valid, resampling!")
    if not valid:
        return False, prompt
    print("Found a valid response!")
    print(response)
    return response, prompt


def get_llama3_prompt(system, user_message):
    prompt = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    return prompt

def run_inference(generator, example, model="llama2"):
    """
    Given the generator and example run inference with the model
    :param generator:
    :param example:
    :param model:
    :return:
    """
    with torch.inference_mode():
        if model == "llama2":
            system = f"<s>[INST]<<SYS>>{DEFAULT_SYSTEM_PROMPT}<</SYS>>\n\n"
            prompt = system + example['instruction'] + "[/INST]"
        elif model == "llama3":
            prompt = get_llama3_prompt(system=DEFAULT_SYSTEM_PROMPT, user_message=example['instruction'])
        response = generator(prompt)
        response = response[0]["generated_text"].replace(prompt, "").strip()
    print(response)
    return response, prompt


def get_completions(output_file, data, generator, model, dataset):
    """
    Get completions for the input file
    :param output_file: outputfile name ends in jsonl
    :param data: json list input data
    :param generator: llama2/llama3 pipeline generator
    :param model: llama2/llama3
    :param dataset: tofueval/ultrachat
    :return:  None
    """
    with open(output_file, "w") as f_:
        for i, t in enumerate(data):
            print(i)
            t = json.loads(t)
            existing_models = t.get('models', [])
            completions = t.get('completions', [])
            try:
                if dataset == "ultrachat":
                    response, prompt = run_inference(generator=generator,
                                             example=t,
                                             model=model)
                elif dataset == "tofueval":
                    response, prompt = get_tofueval_response(generator = generator,
                                                     example = t,
                                                     model = model)
                if not response:
                    print("not writing..")
                    continue
                existing_models.append(model)
                completions.append(response)
                t['initial_response_prompt'] = prompt
                t['models'] = existing_models
                t['completions'] = completions
                data[i] = t
                if i % 100 == 0:
                    print(i)
                f_.write(json.dumps(t))
                f_.write("\n")
            except Exception as e:
                print(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="input jsonl file with the source doc/instructions",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="file to write the completions to; jsonl format",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="token for llama models; can also be set as an environment variable: HF_TOKEN",
    )
    parser.add_argument("--dataset", type=str, help="one of tofueval or utlrachat!")

    args = parser.parse_args()
    print(args)
    model_name = args.model_name
    input_file = args.input_file
    output_file = args.output_file
    hf_token = args.hf_token
    dataset = args.dataset
    os.environ['HF_TOKEN'] = hf_token

    if dataset not in ['tofueval', 'ultrachat']:
        raise ValueError("dataset should be either 'tofueval' or 'ultrachat' ")
    print(input_file)
    print(output_file)

    if ~input_file.endswith("jsonl"):
        raise ValueError("the input path should end with JSONL")
    if ~output_file.endswith("jsonl"):
        raise ValueError("the output path should end with JSONL")

    if dataset == "tofueval":
        if model_name == "llama2":
            model_id = "meta-llama/Llama-2-7b-chat-hf"
            generator = pipeline("text-generation", model=model_id, device_map="auto", max_new_tokens=2048, temperature=0.7)
            print("Loaded llama2!")
        if model_name == "llama3":
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            generator = pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16},
                                 device_map="auto", max_new_tokens=2048, temperature=0.7)

            print("Loaded llama3!")

    elif dataset == "ultrachat":
        if model_name == "llama2":
            model_id = "meta-llama/Llama-2-7b-chat-hf"
            generator = pipeline("text-generation", model=model_id, device_map="auto", max_new_tokens=2048)
            print("Loaded llama2!")
        if model_name == "llama3":  # TODO prompt format for this is not supported yet!
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            generator = pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16},
                                 device_map="auto", max_new_tokens=2048)
            print("Loaded llama3!")

    input_data = open(input_file).readlines()
    get_completions(output_file=output_file,
                    data=input_data,
                    model=model_name,
                    generator=generator,
                    dataset=dataset)

    print("completed!")

if __name__ == "__main__":
    main()
