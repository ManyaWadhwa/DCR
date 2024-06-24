import os
import json
import argparse
import copy
from final_prompts import prompts
from pathlib import Path

system_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
<</SYS>>
{user_prompt}
[/INST]
"""

DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."


def make_llama3_prompt(user_message):
    prompt = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{DEFAULT_SYSTEM_PROMPT}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    return prompt


def get_feedback_values(djson, dataset):
    if dataset == "tofueval":
        input_instruction = djson['source_doc']
        summary = djson['completions'][0]
        aspect = djson['topic']
    else:
        aspect = ""
        instruction_columns = [c for c in ["instruction", "input", "document"] if
                               c in djson]  # find the instruction column
        input_instruction = djson[instruction_columns[0]]  # get the instruction
        summary = djson['completions'][0]  # get the summary
    sentence = djson['sentence']
    inconsistency = djson['inconsistency']
    values = {
        "document": input_instruction,
        "summary": summary,
        "sentence": sentence,
        "span": inconsistency,
        "aspect": aspect,
    }
    return values


def get_refinement_values(djson, dataset):
    if dataset == "tofueval":
        input_instruction = djson['source_doc']
        summary = djson['completions'][0]
        aspect = djson['topic']
        feedback = djson['feedback']
    else:
        aspect = ""
        instruction_columns = [c for c in ["instruction", "input", "document"] if
                               c in djson]  # find the instruction column
        input_instruction = djson[instruction_columns[0]]  # get the instruction
        summary = djson['completions'][0]  # get the summary
        feedback = djson['feedback']
    feedback_str = ""
    for i, feed in enumerate(feedback):
        if 'fix' in feed:
            if len(feed["fix"]) == 0:
                fix = "To fix this, consider removing the information from the summary"
            else:
                fix = f"To fix this, consider changing the span to '{feed['fix']}'"
            feedback_str += f"{i + 1}. For sentence/span in the summary: '{feed['inconsistency']}', feedback is: '{feed['feedback']}'. {fix}\n"
        else:
            print(feed)
            fix = "To fix this, consider removing the information from the summary"
            feedback_str += f"{i + 1}. For sentence/span in the summary: '{feed['inconsistency']}', feedback is: '{feed['feedback']}'. {fix}\n"
    values = {
        "document": input_instruction,
        "summary": summary,
        "aspect": aspect,
        "feedback": feedback_str
    }
    return values


def get_tokens(tokenizer, full_prompt):
    return tokenizer(full_prompt)['input_ids']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=Path,
        help="Path to input file with {instruction}, {completion} {error json} as well as {refinement} keys",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        help="Path to output file",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="model for which to generate instructions -- llama2 or llama3"
    )
    parser.add_argument(
        "--type",
        type=str,
        help="type of training data to create choose from [single_step, feedback_with_correct, feedback_without_correct, refinement_without_correct]"
    )
    parser.add_argument(
        "--filter_long_instances",
        action="store_true",
        default=False,
        help="if filter_long_instance, the code will create a file after filtering all long instances.")
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        help="huggingface model cache dir"
    )

    args = parser.parse_args()
    print(args)

    type = args.type
    if type not in ['single_step', 'two_step_with_minicheck', 'feedback_with_correct', 'feedback_without_correct',
                    'refinement_without_correct']:
        raise ValueError("the type does not match the possible options!")

    assert type in prompts.keys()

    model = args.model
    model_cache_dir = args.model_cache_dir

    if model == "llama2":
        eos_token = "</s>"
    elif model == "llama3":
        eos_token = "<|eot_id|>"
    else:
        raise ValueError(f"{model=} not recognized")

    output_file = args.output_file
    data = open(args.input_file).readlines()
    count = 0
    count_short = 0

    if args.filter_long_instances:
        from transformers import AutoTokenizer
        import os
        base_name = os.path.splitext(output_file)[0]
        print(base_name)
        output_file_short = str(base_name) + "_short.jsonl"
        f_short = open(output_file_short, "w")
        if model == "llama2":
            # model_cache_dir = "/data/users/mwadhwa/models/"
            model_name = "meta-llama/Llama-2-7b-chat-hf"
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
        else:
            # model_cache_dir = "/data/users/mwadhwa/models/"
            model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)

    with open(output_file, "w") as f_:
        print(len(data))
        for i, d in enumerate(data):
            djson = json.loads(d)
            print(i)
            print(djson.keys())
            dataset = djson['dataset']
            if dataset == "tofueval":
                djson['source_prompt'] = copy.deepcopy(djson['prompt'])
                djson['mediasum_dataset_summary'] = copy.deepcopy(djson['summary'])
                summary = djson["completions"][0]
                del djson['summary']
            elif dataset == "ultrachat":
                summary = djson['completions'][0]  # get the summary

            if "feedback" in type:  # if feedback level then
                feedback = djson['feedback']
                assert isinstance(feedback, str)
                fix = djson.get("fix", "")
                label = int(djson['label'])
                values = get_feedback_values(djson, dataset)
                span = values['span']
            else:
                refinement = djson['refinement']
                summary_level_label = djson['summary_factual']
                values = get_refinement_values(djson, dataset)

            prompt = ""
            if args.type == "feedback_with_correct":  # given a span, predict if the span has an error or not
                # TODO ensure the fix is there in the response
                if label == 1:
                    feedback_str = f"no error {eos_token}"  # I guess the issue is there is no
                else:
                    if len(fix) == 0:
                        fix = "To fix this, consider removing the information from the summary."
                    else:
                        fix = f"To fix this, consider changing the span to '{fix}'"
                    assert isinstance(feedback, str)
                    feedback_str = f"{feedback}\nThe error span is: '{span}'\n{fix}{eos_token}"
                prompt = prompts[type][dataset].format(**values)
            elif args.type == "feedback_without_correct":  # given a sentence, extracts the span from the sentence which has an error
                if label == 1:
                    continue
                else:
                    if len(fix) == 0:
                        fix = "To fix this, consider removing the information from the summary."
                    else:
                        fix = f"To fix this, consider changing the span to '{fix}'"
                    assert isinstance(feedback, str)
                    feedback_str = f"{feedback}\nThe error span is: '{span}'\n{fix}{eos_token}"
                prompt = prompts[type][dataset].format(**values)
            elif args.type == "single_step":
                if summary_level_label:
                    feedback_str = f"{summary} {eos_token}"
                else:
                    feedback_str = f"{refinement} {eos_token}"
                prompt = prompts['single_step'][dataset].format(**values)
            elif args.type == "two_step_with_minicheck":
                if summary_level_label:
                    continue
                else:
                    feedback_str = f"{refinement} {eos_token}"
                prompt = prompts[type][dataset].format(**values)
            elif args.type == "refinement_without_correct":
                if summary_level_label:
                    continue
                else:
                    feedback_str = f"{refinement} {eos_token}"
                prompt = prompts['refinement_without_correct'][dataset].format(**values)
            if model == "llama2":
                prompt_with_sys = system_prompt.format(**{"user_prompt": prompt})
            else:
                prompt_with_sys = make_llama3_prompt(user_message=prompt)
            dnew = {
                "id": i,
                "prompt_without_sys": prompt,
                "prompt": prompt_with_sys,
                "response": feedback_str
            }
            djson["prompt"] = prompt_with_sys
            djson["response"] = feedback_str
            print(dnew.keys())
            print(prompt_with_sys)
            print(feedback_str)
            f_.write(json.dumps(dnew))
            f_.write("\n")

            if args.filter_long_instances:
                tokens = get_tokens(tokenizer, djson['prompt'] + djson['response'])
                if len(tokens) < 2010:
                    f_short.write(json.dumps(dnew))
                    f_short.write("\n")
                    count_short += 1

            count += 1
    f_short.close()
    print("written: ", count)
    print("written short: ", count_short)
    print(output_file_short)


if __name__ == "__main__":
    main()
